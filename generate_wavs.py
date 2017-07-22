import os
import pysynth
from piano import Piano

output_folder = "wav"

key_list = Piano().key_list

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for key in key_list:
    print("Generating %s%d" % (key['note'], key['octave']))
    note_str = key['note'].lower() + "%d" % key['octave']
    pysynth.make_wav([[note_str, 4]], fn=os.path.join(output_folder, "%s%d.wav" % (key['note'], key['octave'])))

