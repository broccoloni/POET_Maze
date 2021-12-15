if __name__ == "__main__":
    samples = 5
    reps = [5,5,1,20]
    eps = [0,0.1,0,0]
    for i in range(samples*len(reps)):
        v = i%samples
        ind = i//samples
        path = "job{}.sh".format(i)
        print("making job",i,"path:",path)
        f = open(path,"w+")
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --time=10:00:00\n")
        f.write("#SBATCH --account=def-aali\n")
        f.write("#SBATCH --cpus-per-task=2\n")
        f.write("#SBATCH --mem-per-cpu=32G\n")
        f.write("source ~/lgraha/bin/activate\n")
        f.write("cd ~/scratch/POET_Maze/\n")
        f.write("python runpoet.py {} {:.2f} {}\n".format(v,eps[ind],reps[ind]))
        f.close()
        
