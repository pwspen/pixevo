import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def mutateimg(originalimg, mutatefrac): # Randomizes RGB of fraction of pixels
    img = np.copy(originalimg)
    height, width, chans = img.shape
    numpixels = height * width
    pixels_to_randomize = int(mutatefrac * numpixels)
    if pixels_to_randomize == 0:
        pixels_to_randomize = 1
    random_coordinates = np.random.choice(numpixels, size=pixels_to_randomize, replace=False)
    for pixelnum in random_coordinates:
        coord = all_pixel_coordinates[pixelnum]
        img[coord[1]][coord[0]] = np.random.randint(0, 256, 3)
    return img

def loss(img1, img2): # Calculates absolute difference between all pixel values (all 3 channels) then normalizes into loss
    return np.sum(cv.absdiff(img1, img2))/max_loss

childrendistribution = []
def getchildren(population: int, numchildren: function, numparents: int, mutatefrac: float, victorfac: int, sexualrepr=False) -> dict:
    if childrendistribution == []:
        assert len(population) == numparents

        avgkids = numchildren/numparents
        assert int(avgkids) == avgkids # Children must be integers...





    children = [mutateimg(img, mutatefrac) for x in range(children)]
    losses = [loss(target, child) for child in children]
    kiddict = {loss: kid for kid, loss in zip(children, losses)}
    return kiddict

def evolve(mutatefrac, numchildren, numparents=1, maxgen=np.inf, samelimit=np.inf, victorfac=1, keepbest=False):
    bestloss_alltime = np.inf
    population = [mutateimg(target, 1.0) for x in range(numparents)]
    generation = 0
    losslist = []
    samecounter = 0

    while generation < maxgen:
        kids = getchildren(population, numchildren, numparents, mutatefrac, victorfac) # victorfac is the number of times more children that the best parent has than the worst
        # cull all but top parents
        bestloss = min(kids.keys())
        if bestloss < bestloss_alltime:
            randomimg = kids[bestloss]
            bestloss_alltime = bestloss
            samecounter = 0
        else:
            samecounter += 1
        if samecounter > samelimit:
            if mode == 'single':
                print(f'Hit a wall at gen {generation}, no improvement after {samelimit} gens')
            return (bestloss_alltime, generation)

        losslist.append(bestloss_alltime)
        generation += 1
        if mode == 'single':
            print(f'{generation} {bestloss_alltime:.4f}')
            if (generation % stopevery) == 0:
                concat = np.vstack((target, randomimg))
                cv.namedWindow(displaywindowname)
                cv.setMouseCallback(displaywindowname, mouse_callback)
                cv.imshow(displaywindowname, cv.resize(concat, (0,0), fx=rescale_to_display, fy=rescale_to_display, interpolation=cv.INTER_NEAREST))
                cv.waitKey(0)
    return (bestloss_alltime, generation)


def sweep_mfracs_nkids(mfrange, nkrange, mfpts, nkpts, gens, stop_after_same):
    mfs = np.linspace(*mfrange, mfpts)
    nks = np.linspace(*nkrange, nkpts, dtype=np.int16)
    results = np.zeros((mfpts*nkpts,4))
    for i, mf in enumerate(mfs):
        for j, nk in enumerate(nks):
            results[i*nkpts + j]= (mf, nk, *evolve(mf, nk, gens, stop_after_same)) # _ tosses out generation
            # print(f'MF= {mf:.3f} / NK= {nk:03d} / L= {results[i*nkpts + j, 2]:.4f} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')
            print(f'MF= {mf:.3f} / NK= {nk:03d} / G= {int(results[i*nkpts + j, 3]):04d} / {100*((i*nkpts)+j)/(mfpts*nkpts):2.1f} pct')

    return results

def mouse_callback(event, xc, yc, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.destroyWindow(displaywindowname)
    elif event == cv.EVENT_MBUTTONDOWN:
        cv.destroyWindow(displaywindowname)
        # generations = np.arange(0, len(difflist))
        # plt.scatter(generations, difflist)

        # def hyperbolic(x, a, b):
        #     return a / (b + x)
        # params, _ = curve_fit(hyperbolic, generations, difflist)
        # print(f'A = {params[0]} B = {params[1]}')
        # plt.title('Loss over Generations')
        # plt.xlabel('Generation')
        # plt.ylabel('Loss')
        # plt.show()
        # sys.exit()

if __name__ == "__main__":
    width = 30
    displaywidth = 500
    mode = 'single'

    displaywindowname = 'window'

    img = cv.imread('miss.jpg')
    rescale_to_target = width / np.shape(img)[1]
    target = cv.resize(img, (0, 0), fx=rescale_to_target, fy=rescale_to_target)
    theight, twidth, tchans = target.shape
    max_loss = theight * twidth * tchans * 256
    all_pixel_coordinates = np.array([(x, y) for x in range(target.shape[1]) for y in range(target.shape[0])])
    rescale_to_display = displaywidth / width

    stop_after_same = np.inf
    if mode == 'single':
        mutatefrac = 0.001
        numchildren = lambda k, pop: int((1/k) * pop - 1) # Number of kids as function of success (k=1 is lowest loss of this generation) and pop size
        stopevery = 100 # gens
        evolve(mutatefrac, numchildren, np.inf, stop_after_same)
    elif mode == 'sweep':
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
Function for distributing children. But function has to be stable - produce the right population size, or things will quickly blow up or go to zero.
Easily solved with culling off the end but I don't want to do that. Enticing to have selection mechanics in a single function.
Totally get rid of kid/parent dichotomy
Sexual reproduction
Cladogram mapping and storing - sample every x generations? Species are arbitrary. Need to keep ID of what descended from what.
Can remake IDs every x generations.

"""



    
    