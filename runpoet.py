from POET import *
import sys
almightyint = 6671111 #call pizza pizza hey hey hey

if __name__ == "__main__":
    v = int(sys.argv[1])
    eps = int(sys.argv[2])
    numreps = int(sys.argv[3])
    seed = almightyint + 123456 * v
    print("v:\t{},eps:\t{},numreps:\t{}".format(v,eps,numreps))
    for i in range(500):
        savefreq = 100000 #since were saving here anyways
        poet = POET(v,seed = seed,eps = eps,numreps = numreps,savefreq = savefreq,verbose = True)
        poet.loadlastsave()
        poet.run(20)
        poet.save()
        del poet

