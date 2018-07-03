import os
import sys
import numpy as np
import midi
from PIL import Image
from es_base import ES
from midi_reward import midi_score

input_path = str(sys.argv[1])
output_path = str(sys.argv[2])
populations = int(sys.argv[3])
offsprings = int(sys.argv[4])
generations= int(sys.argv[5])
std_dev = float(sys.argv[6])

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    fullpath = input_path+'/'+filename
    piano_roll = np.array(Image.open(fullpath))
    piano_roll = np.transpose(np.flip(piano_roll, 0), (1,0))
    
    shape = piano_roll.shape
    dna = np.append(piano_roll.flatten(), 127.5).astype(np.float32)[np.newaxis,...].repeat(populations, axis=0)
    pop = dict(DNA=dna, mut_strength=np.random.randn(*dna.shape) * std_dev)
    
    def F(x):
        x = x.copy()
        piano_roll_flattens, thresholds = x[...,:-1], x[...,-1]
        piano_rolls = piano_roll_flattens.reshape(piano_roll_flattens.shape[0],*shape)
        scores = []
        for i, piano_roll in enumerate(piano_rolls):
            piano_roll[piano_roll<thresholds[i]] = 0
            piano_roll[piano_roll>0] = 1
            piano_roll = piano_roll.astype(np.uint8)
            scores.append(midi_score(piano_roll))
        return np.asarray(scores)
    
    es = ES(fitness=F, dna_length=dna.shape[-1], bound=[0, 255], 
        population_size=populations, offspring_size=offsprings, type='maximize')
    for _ in range(generations):
        kids = es.get_offspring(pop)
        pop  = es.put_kids(pop, kids)
        pop  = es.selection(pop)
        print('fitness: %.2f'%F(pop['DNA'][-1:]).mean())
    piano_roll, threshold = pop['DNA'][-1,:-1].reshape(shape), pop['DNA'][-1,-1]
    piano_roll = np.lib.pad(piano_roll, ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
    piano_roll[piano_roll<threshold] = 0
    piano_roll[piano_roll>0] = 255
    piano_roll = piano_roll.astype(np.uint8)

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
