import sys
import numpy as np
import midi
from PIL import Image
import matplotlib.pyplot as plt

input_path = str(sys.argv[1])
output_path = str(sys.argv[2])

piano_roll = np.array(Image.open(input_path))
piano_roll = np.transpose(np.flip(piano_roll, 0), (1,0))
piano_roll = np.lib.pad(piano_roll, ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
# plt.imshow(piano_roll, cmap='gray')
piano_roll[piano_roll<130] = 0
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
midi.write_midifile(output_path, pattern)
