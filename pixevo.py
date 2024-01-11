import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable
from sklearn.metrics import r2_score
from itertools import islice
import time

class PixEvolver():
    def __init__(self, targetfilename, width=20, displaywidth=500, displaywindowname='window', sweep=False):
        self.target = cv2.imread(targetfilename)
        self.width=width
        self.displaywidth=displaywidth
        self.displaywindowname=displaywindowname
        self.rescale_to_target = width / np.shape(self.target)[1]
        self.target = cv2.resize(self.target, (0, 0), fx=self.rescale_to_target, fy=self.rescale_to_target)

        self.theight, self.twidth, self.tchans = self.target.shape
        self.numpixels = self.theight * self.twidth
        self.numchannels = self.numpixels * self.tchans
        self.max_loss = self.theight * self.twidth * self.tchans * 256
        self.all_pixel_coordinates = np.array([(y, x) for x in range(self.target.shape[1]) for y in range(self.target.shape[0])])
        self.all_channel_coordinates = np.array([(y, x, z) for x in range(self.target.shape[1]) for y in range(self.target.shape[0]) for z in range(3)])
        self.rescale_to_display = displaywidth / width
        self.stop_after_same = np.inf
        self.sweep = sweep

        if not self.sweep:
            mutatefrac = lambda loss: 0.004 * loss
            morekidsfac = 1
            childrenbasefunction = lambda k, pop: 3
            # childrenbasefunction = lambda k, pop: int((1/k) * pop - 1) if k != 0 else int(pop - 1) # Number of kids as function of success (k=1 is lowest loss of this generation) and pop size
            numchildren = lambda k, pop: int(morekidsfac * childrenbasefunction(k, pop))
            numparents = 10 # reproducers every generation
            maxgen = np.inf
            samelimit = 10000 # gens
            keepbest = False
            sexualrepr = False
            updatecycle = 20 # gens
            
            self.evolvesettings(mutatefrac, numchildren, numparents, maxgen, samelimit, keepbest, sexualrepr, updatecycle)
            self.evolveprep()

        else:
            mfracrange = [0.1, 0.001]
            nkidsrange = [100, 5]
            gens = np.inf
            pts = 20
            cap = 100 # maximum number alive
            results = self.sweep_mfracs_nkids(mfracrange, nkidsrange, pts, pts, gens, self.stop_after_same)
            mfs = results[:, 0]
            nks = results[:, 1]
            loss = results[:, 2]
            gens = results[:, 3]

            plt.scatter(mfs, nks, c=gens, cmap='hot')
            plt.colorbar(label='Gens until stop')
            plt.xlabel('Mutation Fraction')
            plt.ylabel('Number of Children')

            plt.show()

    def mutateimg(self, parent1, mutatefrac, parent2=None): # Randomizes RGB of fraction of pixels
        channels_to_randomize = int(mutatefrac * self.numchannels)
        parent1 = np.copy(parent1)
        if parent2 is not None:
            parent2 = np.copy(parent2)
            if parent1.shape != parent2.shape:
                raise ValueError('Parent images must be the same shape') 
            px_from_parent2 = int(0.5 * self.numpixels)
            parent2_coordinates = np.random.choice(self.numpixels, size=px_from_parent2, replace=False)
            for px in parent2_coordinates:
                parent1[self.all_pixel_coordinates[px]] = parent2[self.all_pixel_coordinates[px]]

        if channels_to_randomize == 0:
            channels_to_randomize = 1
        random_coordinates = np.random.choice(self.numchannels, size=channels_to_randomize, replace=False)

        mutmag = mutatefrac / (1/self.numchannels)
        mutmag = 256 if mutmag > 1 else int(256 * mutmag) + 3 # 3 is to improve performance at end of long tail very close to 0 loss

        for channum in random_coordinates:
            coord = self.all_channel_coordinates[channum]
            mag = np.random.randint(-mutmag/2, mutmag/2) + 256 
            parent1[*coord] += mag # makes use of uint8 wraparound
            # print(coord)
        return parent1

    def calculate_loss(self, img1, img2): # Calculates absolute difference between all pixel values (all 3 channels) then normalizes into loss
        return np.sum(cv2.absdiff(img1, img2))/self.max_loss

    def getchildren(self, mutatefrac: Callable[[float], float], childrendistribution: dict, population: dict=None, sexualrepr: bool=False) -> dict:
        """
        numchildren: Callable[[int], list] or int
            If Callable, returns list of children for each parent
            If int, returns that many children for each parent
        mutatefrac: float
            Fraction of pixels to mutate
        population: dict
            Dictionary of parents
        sexualrepr: bool
            Whether to use sexual reproduction
        single lineage mode: numchildren returns int, population contains a single image: {1: img}
        """
        if sexualrepr and len(population) < 2:
            raise ValueError('Population must be at least 2 for sexual reproduction')
        if sexualrepr and len(population) % 2 != 0:
            raise ValueError('Population must be even for sexual reproduction')
        
        children = {}
        if sexualrepr:
            for k in range(1, len(population), 2):
                children[k] = {i: self.mutateimg(parent1=population[k]["img"], parent2=population[k+1]["img"], mutatefrac=mutatefrac(population[k]["loss"])) for i in range(childrendistribution[k])}
                children[k+1] = {i: self.mutateimg(parent1=population[k]["img"], parent2=population[k+1]["img"], mutatefrac=mutatefrac(population[k+1]["loss"])) for i in range(childrendistribution[k+1])}
                children[k]["inheritance"] = (population[k]["inheritance"], population[k+1]["inheritance"])
                children[k+1]["inheritance"] = (population[k]["inheritance"], population[k+1]["inheritance"])

        else:
            for k, kdict in population.items():
                children[k] = {i: self.mutateimg(kdict["img"], mutatefrac(kdict["loss"])) for i in range(childrendistribution[k])}
                children[k]["inheritance"] = kdict["inheritance"]

        # Calculate loss for each child
        img_with_loss = []
        for child in children.values():
            inher = child["inheritance"]
            for key, val in child.items():
                if key != "inheritance":
                    image = val
                    loss = self.calculate_loss(image, self.target)
                    img_with_loss.append((image, loss, inher))

        # Sort all images by loss
        img_with_loss.sort(key=lambda x: x[1])

        # Reconstruct the dictionary with sorted images
        kiddict = {}
        for img, loss, inher in img_with_loss:
            kiddict[len(kiddict) + 1] = {"img": img, "loss": loss, "inheritance": inher}

        return kiddict

    def evolvesettings(self, mutatefrac: Callable[[float], float], numchildren: Callable, numparents: int=1, maxgen=np.inf, samelimit=np.inf, keepbest=False, sexualrepr=False, updatecycle: int=100):
        if sexualrepr and numparents < 2:
            raise ValueError('Sexual reproduction requires at least 2 parents')
        if sexualrepr and numparents % 2 != 0:
            raise ValueError('Sexual reproduction requires an even number of parents')
        
        self.mutatefrac = mutatefrac
        self.numchildren = numchildren
        self.numparents = numparents
        self.maxgen = maxgen
        self.samelimit = samelimit
        self.keepbest = keepbest
        self.sexualrepr = sexualrepr
        self.updatecycle = updatecycle
        self.childrendistribution = {k+1: numchildren(k, numparents) for k in range(numparents)}

    def evolveprep(self):
        self.inheritances = {}
        self.bestloss_alltime = np.inf
        self.generation = 0
        self.samecounter = 0
        self.losslist = []
        self.population = {k + 1: {"img": self.mutateimg(self.target, 1.0), "loss": 1.0, "inheritance": k + 1} for k in range(self.numparents)}

    def evolvestats(self):
        print('childrendistribution', list(self.childrendistribution.values()))
        print('numparents', self.numparents)
        print('numchildren', sum(self.childrendistribution.values()))
        print(f'pixels', self.numpixels)
        print(f'equivalent genome size {4 * self.numchannels} BP')
        print(f'mutatefrac floor {1/self.numpixels:.4f}')

    def evolve(self, generations, display=False):
        self.end = self.generation + generations
        if display:
            cv2.namedWindow(self.displaywindowname)
            clickhandler = ClickEventHandler(self.losslist, self.displaywindowname)

        while self.generation < self.end: # main evolutionary loop
            if display:
                if clickhandler.pause:
                    cv2.waitKey(100)
                    continue
            prebestloss = self.population[1]["loss"]
            kids = self.getchildren(population=self.population, mutatefrac=self.mutatefrac,
                                    sexualrepr=self.sexualrepr, childrendistribution=self.childrendistribution)
            self.population = dict(islice(kids.items(), self.numparents)) # cull population to numparents
            try:
                bestloss, bestimg = self.population[0]["loss"], self.population[0]["img"]
            except KeyError:
                bestloss, bestimg = self.population[1]["loss"], self.population[1]["img"]

            if bestloss < prebestloss:
                samecounter = 0
                bestloss_alltime = bestloss
                if self.keepbest:
                    self.population[0] = bestimg # insert best image into population (key 0 is never set by getchildren)
            else:
                samecounter += 1
            if samecounter > self.samelimit:
                print(f'Hit a wall at gen {self.generation}, no improvement after {self.samelimit} gens')
                return {'bestloss': bestloss_alltime, 'generation': self.generation}

            self.losslist.append(bestloss_alltime)
            if display:
                concat = np.hstack((self.target, bestimg)) if self.theight > self.twidth else np.vstack((self.target, bestimg))
                cv2.imshow(self.displaywindowname, cv2.resize(concat, (0,0), fx=self.rescale_to_display, fy=self.rescale_to_display, interpolation=cv2.INTER_NEAREST))
                cv2.setMouseCallback(self.displaywindowname, clickhandler.handle_click)
                cv2.waitKey(1)

            if self.generation % self.updatecycle == 0:
                chansrandomized = self.numchannels * self.mutatefrac(bestloss_alltime)
                self.inheritances[self.generation] = [self.population[k]["inheritance"] for k in self.population.keys()]
                if display:
                    print(f'GEN={self.generation} LOSS={bestloss_alltime:.4f} '
                        f'{f"CHANS={int(chansrandomized)} " if chansrandomized > 1 else f"RGB +/- {256 * chansrandomized:.4f} "}'
                        f"ABSDIFF={bestloss_alltime * self.max_loss:.0f} "
                        f"UNIQUE={len(set(self.inheritances[self.generation]))}")

                for k in self.population.keys():
                    self.population[k]["inheritance"] = k
            self.generation += 1

        return {'bestloss': bestloss_alltime, 'generation': self.generation}

    def sweep_mfracs_nkids(self, mfrange, nkrange, mfpts, nkpts, gens, stop_after_same):
        mfs = np.linspace(*mfrange, mfpts)
        nks = np.linspace(*nkrange, nkpts, dtype=np.int16)
        results = np.zeros((mfpts*nkpts,4))
        for i, mf in enumerate(mfs):
            for j, nk in enumerate(nks):
                results[i*nkpts + j]= (mf, nk, *self.evolve(mutatefrac=mf, numchildren=lambda: nk, 
                                                    maxgen=gens, samelimit=stop_after_same)["generation"])
                # print(f'MF= {mf:.3f} / NK= {nk:03d} / L= {results[i*nkpts + j, 2]:.4f} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')
                print(f'MF= {mf:.3f} / NK= {nk:03d} / G= {int(results[i*nkpts + j, 3]):04d} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')

        return results

class ClickEventHandler:
            def __init__(self, losslist, displaywindowname):
                self.losslist = losslist # reference to main losslist
                self.pause = False
                self.displaywindowname = displaywindowname

            def handle_click(self, event, x, y, flags, param):
                if flags == cv2.EVENT_FLAG_LBUTTON:
                    self.pause = not self.pause
                elif flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_SHIFTKEY:
                    cv2.destroyWindow(self.displaywindowname)
                    sys.exit()
                elif flags == cv2.EVENT_FLAG_MBUTTON:
                    generations = np.arange(0, len(self.losslist))
                    plt.scatter(generations, np.array(self.losslist), marker='o', c="r", label="data")

                    def hyperbolic(x, a, b, c):
                        return (a / (x + b)) + c

                    def power(x, a, b, c):
                        return a * (x ** b) + c
                    
                    params, _ = curve_fit(hyperbolic, generations, np.array(self.losslist))
                    predicteddata = hyperbolic(generations, *params)
                    r2 = r2_score(self.losslist, predicteddata)
                    asymptote = hyperbolic(np.inf, *params)
                    plt.plot(generations, predicteddata, marker='x', c="b", label=f'predicted\nR2={r2:.3f}\nasymptote={asymptote:.3f}')
                    plt.legend()
                    plt.xlabel('Generation')
                    plt.ylabel('Loss')
                    plt.show()

pe = PixEvolver(targetfilename='miss.jpg')
pe.evolve(generations=1000, display=True)
time.sleep(3)
pe.evolve(generations=1000, display=True)
time.sleep(3)
pe.evolve(generations=1000, display=True)



"""
Future work:
How to encourage different lineages to stay around? Only a single niche is available
Each image is a different niche?


Mutations:
Add different types of mutations. Changing mutation from pixel to channel improved speed a LOT.
Large mutations improve initial performance, small mutations improve long-stretch performance.
Difficult for a mutation to improve performance when it has to get increasingly lucky over time, so assume state is already pretty close to optimal.
Mutation list: 
    Add random number (-128 to 128) to channel value. range decreases with loss
    Location-centered mutation - location of randomization is weighted towards random coordinates (will this even do anything idk)
    Color-centered mutation - color of randomization is weighted towards number (same)


Improve performance of sexual reproduction

Channel reroll: Too random for long tail optimization
Channel addition: Range gets too small too fast so many wrong colors remain when ability to change is low (turns out loss tuning was just wrong)
Need to save loss curve and initial stats for comparison

"""

"""
Unexpected stuff:
Loss curve depends pretty heavily on image, even when rescaled to same size!
Needs smaller mutations to get high level of detail
Gets down to small but nonzero loss, different each run.. why? never leaves minima
"""