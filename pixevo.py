import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable
from sklearn.metrics import r2_score
from itertools import islice
import time

def mutateimg(originalimg, mutatefrac): # Randomizes RGB of fraction of pixels
    img = np.copy(originalimg)
    height, width, chans = img.shape
    numpixels = height * width
    numchannels = numpixels * 3
    channels_to_randomize = int(mutatefrac * numchannels)
    if channels_to_randomize == 0:
        channels_to_randomize = 1
    random_coordinates = np.random.choice(numchannels, size=channels_to_randomize, replace=False)
    mutmag = mutatefrac / (1/numchannels)
    mutmag = 256 if mutmag > 1 else int(256 * mutmag)
    for channum in random_coordinates:
        coord = all_channel_coordinates[channum]
        img[coord[1]][coord[0]][coord[2]] += np.random.randint(-mutmag/2, mutmag/2)
    return img

def calculate_loss(img1, img2): # Calculates absolute difference between all pixel values (all 3 channels) then normalizes into loss
    return np.sum(cv2.absdiff(img1, img2))/max_loss

def getchildren(numchildren: Callable[[int], list], mutatefrac: Callable[[float], float], childrendistribution: dict, population: dict=None, sexualrepr: bool=False) -> dict:
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
    children = {}
    for k, kdict in population.items():
        children[k] = [mutateimg(kdict["img"], mutatefrac(kdict["loss"])) for _ in range(childrendistribution[k])]

    # Calculate loss for each child
    img_with_loss = []
    for key, images in children.items():
        for img in images:
            loss = calculate_loss(img, target)
            img_with_loss.append((img, loss))

    # Sort all images by loss
    img_with_loss.sort(key=lambda x: x[1])

    # Reconstruct the dictionary with sorted images
    kiddict = {}
    for img, loss in img_with_loss:
        kiddict[len(kiddict) + 1] = {"img": img, "loss": loss}

    return kiddict

def evolve(mutatefrac: Callable[[float], float], numchildren: Callable, numparents: int=1, maxgen=np.inf, samelimit=np.inf, keepbest=False, sexualrepr=False, singlelinmode=False, showevery: int=100):
    if singlelinmode and numparents != 1:
        raise ValueError('Single lineage mode only supports one parent')
    elif not singlelinmode and numparents == 1:
        raise ValueError('Only single lineage mode supports one parent')
    if sexualrepr and numparents < 2:
        raise ValueError('Sexual reproduction requires at least 2 parents')
    if sexualrepr and numparents % 2 != 0:
        raise ValueError('Sexual reproduction requires an even number of parents')
    bestloss_alltime = np.inf
    generation = 0
    samecounter = 0
    losslist = []
    population = {k + 1: {"img": mutateimg(target, 1.0), "loss": 1.0} for k in range(numparents)}

    class ClickEventHandler:
        def __init__(self):
            self.last_click_time = 0
            self.double_click_flag = False
            self.double_click_threshold = 0.3  # 300ms, adjust as needed

        def handle_click(self, event, x, y, flags, param):
            if flags == cv2.EVENT_FLAG_LBUTTON:
                cv2.destroyWindow(displaywindowname)
            elif flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_SHIFTKEY:
                cv2.destroyWindow(displaywindowname)
                sys.exit()
            elif flags == cv2.EVENT_FLAG_MBUTTON:
                generations = np.arange(0, len(losslist))
                plt.scatter(generations, np.array(losslist), marker='o', c="r", label="data")

                def hyperbolic(x, a, b, c):
                    return (a / (x + b)) + c
                
                params, _ = curve_fit(hyperbolic, generations, np.array(losslist))
                predicteddata = hyperbolic(generations, *params)
                r2 = r2_score(losslist, predicteddata)
                asymptote = hyperbolic(np.inf, *params)
                plt.plot(generations, predicteddata, marker='x', c="b", label=f'predicted\nR2={r2:.3f}\nasymptote={asymptote:.3f}')
                plt.legend()
                plt.xlabel('Generation')
                plt.ylabel('Loss')
                plt.show()

    childrendistribution = {k: numchildren(k, len(population)) for k in population.keys()}
    print('childrendistribution', list(childrendistribution.values()))
    print('numparents', numparents)
    print('numchildren', sum(childrendistribution.values()))
    print(f'pixels', int(theight*twidth))
    print(f'equivalent genome size {int((theight*twidth*3*256)**(1/4))} - {int((theight*twidth*3*256)/4)} BP') # would be /4, except order matters for genomes, not for images.
    print(f'mutatefrac floor {1/int(theight*twidth):.4f}')

    concat = np.hstack((target, target))
    cv2.namedWindow(displaywindowname)
    clickhandler = ClickEventHandler()
    cv2.setMouseCallback(displaywindowname, clickhandler.handle_click)
    cv2.imshow(displaywindowname, cv2.resize(concat, (0,0), fx=rescale_to_display, fy=rescale_to_display, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(1)

    while generation < maxgen:
        prebestloss = population[1]["loss"]
        population = getchildren(population=population, numchildren=numchildren, mutatefrac=mutatefrac,
                                  sexualrepr=sexualrepr, childrendistribution=childrendistribution)
        population = dict(islice(population.items(), numparents)) # cull population to numparents
        bestloss, bestimg = population[1]["loss"], population[1]["img"]

        if bestloss < prebestloss:
            samecounter = 0
            bestloss_alltime = bestloss
            if keepbest:
                population[0] = bestimg # insert best image into population (key 0 is never set by getchildren)
        else:
            samecounter += 1
        if samecounter > samelimit:
            print(f'Hit a wall at gen {generation}, no improvement after {samelimit} gens')
            return {'bestloss': bestloss_alltime, 'generation': generation}

        losslist.append(bestloss_alltime)

        concat = np.hstack((target, bestimg))
        cv2.imshow(displaywindowname, cv2.resize(concat, (0,0), fx=rescale_to_display, fy=rescale_to_display, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)

        if generation % int(showevery/100) == 0:
            print(f'{generation} {bestloss_alltime:.4f} {int(3 * theight * twidth * mutatefrac(bestloss_alltime))}')
        generation += 1
        # if (generation % showevery) == 0:
        #     concat = np.vstack((target, bestimg))
        #     cv2.namedWindow(displaywindowname)
        #     clickhandler = ClickEventHandler()
        #     cv2.setMouseCallback(displaywindowname, clickhandler.handle_click)
        #     cv2.imshow(displaywindowname, cv2.resize(concat, (0,0), fx=rescale_to_display, fy=rescale_to_display, interpolation=cv2.INTER_NEAREST))
        #     cv2.waitKey(0)
    return {'bestloss': bestloss_alltime, 'generation': generation}

def sweep_mfracs_nkids(mfrange, nkrange, mfpts, nkpts, gens, stop_after_same):
    mfs = np.linspace(*mfrange, mfpts)
    nks = np.linspace(*nkrange, nkpts, dtype=np.int16)
    results = np.zeros((mfpts*nkpts,4))
    for i, mf in enumerate(mfs):
        for j, nk in enumerate(nks):
            results[i*nkpts + j]= (mf, nk, *evolve(mutatefrac=mf, numchildren=lambda: nk, 
                                                   maxgen=gens, samelimit=stop_after_same, singlelinmode=True)["generation"])
            # print(f'MF= {mf:.3f} / NK= {nk:03d} / L= {results[i*nkpts + j, 2]:.4f} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')
            print(f'MF= {mf:.3f} / NK= {nk:03d} / G= {int(results[i*nkpts + j, 3]):04d} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')

    return results

if __name__ == "__main__":
    width = 30
    displaywidth = 500
    sweep = False
    displaywindowname = 'window'

    img = cv2.imread('sunset.jpg')
    rescale_to_target = width / np.shape(img)[1]
    target = cv2.resize(img, (0, 0), fx=rescale_to_target, fy=rescale_to_target)
    theight, twidth, tchans = target.shape
    max_loss = theight * twidth * tchans * 256
    all_channel_coordinates = np.array([(x, y, z) for x in range(target.shape[1]) for y in range(target.shape[0]) for z in range(3)])
    rescale_to_display = displaywidth / width

    stop_after_same = np.inf
    if not sweep:
        mutatefrac = lambda loss: 0.001 * loss
        morekidsfac = 1
        # childrenbasefunction = lambda k, pop: 3
        childrenbasefunction = lambda k, pop: int((1/k) * pop - 1) if k != 0 else int(pop - 1) # Number of kids as function of success (k=1 is lowest loss of this generation) and pop size
        numchildren = lambda k, pop: int(morekidsfac * childrenbasefunction(k, pop))
        numparents = 20 # reproducers every generation
        maxgen = np.inf
        samelimit = 100 # gens
        keepbest = False
        sexualrepr = False
        singlelinmode = False
        stopevery = 5000 # gens
        results = evolve(mutatefrac=mutatefrac, numchildren=numchildren, numparents=numparents,
                         maxgen=maxgen, samelimit=samelimit, keepbest=keepbest, sexualrepr=sexualrepr,
                           singlelinmode=singlelinmode, showevery=stopevery)
    else:
        mfracrange = [0.1, 0.001]
        nkidsrange = [100, 5]
        gens = np.inf
        pts = 20
        cap = 100 # maximum number alive
        results = sweep_mfracs_nkids(mfracrange, nkidsrange, pts, pts, gens, stop_after_same)
        mfs = results[:, 0]
        nks = results[:, 1]
        loss = results[:, 2]
        gens = results[:, 3]
        plt.scatter(mfs, nks, c=gens, cmap='hot')
        plt.colorbar(label='Gens until stop')
        plt.xlabel('Mutation Fraction')
        plt.ylabel('Number of Children')

        plt.show()

"""
Future work:
Add different types of mutations. Changing mutation from pixel to channel improved speed a LOT.
Large mutations improve initial performance, small mutations improve long-stretch performance.
Difficult for a mutation to improve performance when it has to get increasingly lucky over time, so assume state is already pretty close to optimal.
Mutation list: 
    Add random number (-128 to 128) to channel value. range decreases with loss
    Location-centered mutation - location of randomization is weighted towards random coordinates (will this even do anything idk)
    Color-centered mutation - color of randomization is weighted towards number (same)


Sexual reproduction
Cladogram mapping and storing - sample every x generations? Species are arbitrary. Need to keep ID of what descended from what.
Can remake IDs every x generations.

"""

"""
Unexpected stuff:
Loss curve depends pretty heavily on image, even when rescaled to same size!
Needs smaller mutations to get high level of detail
"""