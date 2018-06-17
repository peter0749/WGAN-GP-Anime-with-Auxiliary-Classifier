import sys
import os
import numpy as np
import scipy
import midi
from PIL import Image

lower_bound=34
upper_bound=97
bounded=upper_bound-lower_bound+1 # 64
segLen=96
hopLen=96

def Tempo2BPM(x):
    ret = x.data[2] | x.data[1]<<8 | x.data[0]<<16
    ret = float(60000000)/float(ret)
    return ret

prefix = str(sys.argv[1])
output_dir = str(sys.argv[2])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(prefix):
    fullfile = prefix+'/'+str(filename)
    try:
        midi_in = midi.read_midifile(fullfile)
    except:
        continue
    sheet_grid = np.zeros((segLen,bounded), dtype=np.float32)
    for track in midi_in:
        Normal = 120.0
        defaultRes = 4
        ResScale = float(midi_in.resolution) / float(defaultRes)
        speedRatio = 1.0
        accumTick = 0.
        firstChange = True
        swLine = np.zeros(bounded, dtype=np.float32)
        lastTick = 0.
        for v in track:
            if isinstance(v, midi.ProgramChangeEvent) and v.data[0]!=0:
                break
            if hasattr(v, 'tick') :
                tick = float(v.tick)/speedRatio/ResScale
                accumTick += tick
                accumTick_i = int(accumTick)
                lastTick_i = int(lastTick)
                if accumTick_i>lastTick_i:
                    if sheet_grid.shape[0]<accumTick_i:
                        sheet_grid = np.lib.pad(sheet_grid, ((0,accumTick_i-lastTick_i),(0,0)), 'constant', constant_values=((0,0),(0,0)))
                    new = sheet_grid[lastTick_i:accumTick_i] + swLine
                    sheet_grid[lastTick_i:accumTick_i,:] = new
                    if (isinstance(v, midi.NoteOffEvent) or isinstance(v, midi.NoteOnEvent)) and v.tick==1 and v.data[0]>=lower_bound and v.data[0]<=upper_bound:
                        sheet_grid[max(lastTick_i,accumTick_i-1), v.data[0]-lower_bound] = 0
                    lastTick = accumTick
            if isinstance(v, midi.SetTempoEvent):
                changeBPM = Tempo2BPM(v)
                if firstChange:
                    firstChange = False
                    Normal = changeBPM
                    continue
                speedRatio = float(changeBPM)/float(Normal)
            elif isinstance(v, midi.NoteOnEvent) and v.data[0]>=lower_bound and v.data[0]<=upper_bound and v.data[1]>0:
                swLine[v.data[0]-lower_bound] = v.data[1]
            elif (isinstance(v, midi.NoteOffEvent) or (isinstance(v, midi.NoteOnEvent) and v.data[1]==0)) and v.data[0]>=lower_bound and v.data[0]<=upper_bound:
                swLine[v.data[0]-lower_bound] = 0
    sheet_grid[sheet_grid>0] = 255
    if np.sum(sheet_grid)==0: continue
    while len(sheet_grid)>0 and np.sum(sheet_grid[-1,:])==0:
        sheet_grid= np.delete(sheet_grid, -1, 0) ## trim 0
    sheet_grid = np.flip(np.transpose(np.clip(sheet_grid,0,255).astype(np.uint8), (1,0)), 0)
    for i in range(0, sheet_grid.shape[1]-segLen, hopLen): # alignment
        note_ratio = float(np.sum(sheet_grid[:,i:i+segLen]>0)) / float(segLen*60)
        if note_ratio < 0.0005: continue
        im = Image.fromarray(sheet_grid[:,i:i+segLen])
        saveto = output_dir+'/'+''.join(filename.split('.')[:-1])+'_p'+str(i)+'.jpg'
        im.save(saveto, format='JPEG', subsampling=0, quality=100)
        print(('image saved to: ' + saveto))
