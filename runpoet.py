from POET import *
import sys
almightyint = 6671111 #call pizza pizza hey hey hey

if __name__ == "__main__":
    v = int(sys.argv[1])
    eps = float(sys.argv[2])
    numactions = int(sys.argv[3])
    seed = almightyint + 123456 * v
    print("v:\t{},eps:\t{:.1f},numactions:\t{}".format(v,eps,numactions))
    for i in range(500):
        poet = POET(v,seed = seed,eps = eps,numactions = numactions,verbose = True)
        poet.loadlastsave()
        poet.run(20)
        poet.save()
        del poet

