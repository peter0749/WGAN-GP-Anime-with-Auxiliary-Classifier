import os
import sys
import numpy as np
import midi
from PIL import Image

input_path = str(sys.argv[1])
output_path = str(sys.argv[2])
threshold = int(sys.argv[3])

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    fullpath = input_path+'/'+filename
    piano_roll = np.array(Image.open(fullpath))
    piano_roll = np.transpose(np.flip(piano_roll, 0), (1,0))
    piano_roll = np.lib.pad(piano_roll, ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
    piano_roll[piano_roll<threshold] = 0
    piano_roll[piano_roll>0] = 255

    pattern = midi.Pattern(resolution=4)
    track = midi.Track()
    pattern.append(track)
    lastEvent = 0
    sweep = np.zeros_like(piano_roll[0], dtype=np.bool)
    for tick, line in enumerate(piano_roll):
        for i, note in enumerate(line):
            if note>0 and not sweep[i]: # a note on event
                track.append(midi.NoteOnEvent(tick=tick-lastEvent, data=[i+34 ,127]))
                lastEvent = tick
                sweep[i]=True
            if note==0 and sweep[i]: # a note off event
                track.append(midi.NoteOffEvent(tick=tick-lastEvent, data=[i+34 ,0]))
                lastEvent = tick
                sweep[i]=False
    track.append(midi.EndOfTrackEvent(tick=1))
    midi.write_midifile(output_path+'/'+filename+'.mid', pattern)
