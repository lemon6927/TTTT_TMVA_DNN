#!/usr/bin/env python
import glob, os, math
import varsList
import datetime

outPath = os.getcwd()
outFile = os.listdir(outPath+'/condor_log/')
seedOutDirectory = [seedStr for seedStr in outFile if ".out" in seedStr]

numVars = int(seedOutDirectory[0].split("vars_")[0].split("Keras_")[1])

varImportance_file = open(outPath+'/dataset/VariableImportanceResults_' + str(numVars) + 'vars.txt','w')

varImportance_file.write("Number of Variables: {}, Date: {}".format(numVars,datetime.datetime.today().strftime('%Y-%m-%d')))

seedList = []
for Files in outFile:
  if "Subseed_" not in Files and ".out" in Files:
    seedList.append(Files.split("_Seed_")[1].split(".out")[0])

os.chdir(outPath+'/condor_log')

seedDict = {}
for index, seed in enumerate(seedList):
  if index > 500: break
  seedDict[seed] = glob.glob("Keras_" + str(numVars) + "vars_Seed_" + seed + "_Subseed_*.out")
 
importances = {}
for index in range(0,numVars):
  importances[index] = 0
  
for seeds in seedDict:
  l_seed = long(seeds)
  varImportance_file.write("\nSeed: {}, Subseeds: ".format(seeds))
  for line in open("Keras_" + str(numVars) + "vars_Seed_" + seeds + ".out").readlines():
    if "ROC-integral" in line:
      SROC = float(line[:-1].split(' ')[-1][:-1])
  for subseedout in seedDict[seeds]:
    subseed = subseedout.split("_Subseed_")[1].split(".out")[0]
    l_subseed = long(subseed)
    varImportance_file.write("{} ".format(subseed))
    varIndx = math.log(l_seed - l_subseed)/0.693147
    for line in open("Keras_" + str(numVars) + "vars_Seed_" + seeds + "_Subseed_" + subseed + ".out").readlines():
      if "ROC-integral" in line:
        SSROC = float(line[:-1].split(" ")[-1][:-1])
        importances[int(varIndx)] += SROC - SSROC
        
normalization = np.sum(importances)
  
varImportance_file.write("\nImportance calculation:")
varImportance_file.write("\nNormalization: {}".format(normalization))
varImportance_file.write("\n{:32}, {:5}:".format(Variable,Importance))

for index in range(0,numVars):
  varImportance_file.write("\n{:32}, {:5}".format(varsList.varList["BigComb"][index][0],100*importances[index]/normalization))

varImportance_file.close()
print("Finished variable importance calculation for {} variables.".format(numVars))
