#!/usr/bin/env th

require 'torch'
require 'optim'

require 'paths'

require 'xlua'
require 'csvigo'

require 'nn'
require 'dpnn'

require 'io'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.device)
end

opt.manualSeed = 2
torch.manualSeed(opt.manualSeed)

paths.dofile('dataset.lua')
paths.dofile('banana-represent.lua')

model = torch.load(opt.model)
model:evaluate()
if opt.cuda then
   model:cuda()
end

--csv = csvigo.load({path = "./reps.csv", verbose = false, mode = "raw"})
--banana_repsCSV = csvigo.load({path = "./batch-represent/reps.csv", verbose = false, mode = "raw"})
banana_labelsCSV = csvigo.load({path = "./data/mydataset/banana_feature/labels.csv", verbose = false, mode = "raw"})
--print(banana_labelsCSV)
preTotalImg = table.getn(banana_labelsCSV)
print (("available images: %d"):format(preTotalImg))
preTotalLabel = (banana_labelsCSV[preTotalImg][1])--column,row
print (("available labels: %d"):format(preTotalLabel))

start_time = os.time()
--newCSV = csvigo.File(paths.concat(opt.outDir, "new.csv"), 'w')
repsCSV = csvigo.File(paths.concat(opt.outDir, "reps.csv"), 'w')
labelsCSV = csvigo.File(paths.concat(opt.outDir, "labels.csv"), 'w')
preTotalLabel = preTotalLabel + 1
batchRepresent()

labelsCSV:writeall(banana_labelsCSV)

end_time = os.time()

dt = os.difftime(end_time-start_time)
print(('classification took: %d secs'):format(dt))

repsCSV:close()
labelsCSV:close()
--newCSV:close()
