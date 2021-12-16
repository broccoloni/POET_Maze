from POET import *
import sys
import timeit
almightyint = 6671111 #call pizza pizza hey hey hey

if __name__ == "__main__":
    v = int(sys.argv[1])
    eps = int(sys.argv[2])
    numreps = int(sys.argv[3])
    seed = almightyint + 123456 * v
    print("v:\t{},eps:\t{},numreps:\t{}".format(v,eps,numreps))
    start = timeit.default_timer()
    savefreq = 5
    for i in range(500):
        runningtime = timeit.default_timer() - start
        if runningtime > 9*60*60:
            savefreq = 1
        savefreq = 100000 #since were saving here anyways
        poet = POET(v,seed = seed,eps = eps,numreps = numreps,savefreq = savefreq,verbose = True)
        poet.loadlastsave()
        poet.run(savefreq)
        poet.save()
        del poet

