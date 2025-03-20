'''
Author: Janni Nikolaides
Date: 2025-01-28
Description: This script is used to check for cosmic ray contamination in IceCube data. 
It reads in a set of I3 files and checks for cosmic ray contamination in each event. 
If an event is clean, it saves the information in the header file (Event ID, Run ID etc.) to a csv file. 
The script is parallelized to speed up the process.
Usage: Change base_dir and set range to number of subdirectories in base_dir. Don't forget to change the output file name.
'''

from icecube import icetray, dataio, dataclasses, simclasses, phys_services, clsim
import csv
from concurrent.futures import ProcessPoolExecutor
import os

def iterate_and_find_ids(tree, prim):
    ids = [prim.id]
    for d in tree.children(prim.id):
        ids.extend(iterate_and_find_ids(tree, d))
    return ids

def check_cr_contamination(frame, i3mctree_name, allowed_max_bg=0):

    tree = frame[i3mctree_name]

    prim_index = 0
    found = False
    for p in tree.primaries:
        if p.type in {p.NuE, p.NuEBar, p.NuMu, p.NuMuBar, p.NuTau, p.NuTauBar}:
            found = True
            break
        prim_index += 1

    assert prim_index == 0
    assert found

    bg_ids = set()
    signal_ids = set()

    for cur_prim_index, cur_prim in enumerate(tree.primaries):
        if cur_prim_index == 0:
            signal_ids.update(iterate_and_find_ids(tree, cur_prim))
        else:
            bg_ids.update(iterate_and_find_ids(tree, cur_prim))

    mcpe_particle_id_map = frame["I3MCPESeriesMapParticleIDMap"]

    num_sig = 0
    num_bg = 0
    for k, id_map in mcpe_particle_id_map.items():
        for id_key, hits in id_map.items():
            if id_key in signal_ids:
                num_sig += len(hits)
            elif id_key in bg_ids:
                num_bg += len(hits)
            else:
                raise Exception()

    return num_bg <= allowed_max_bg and num_sig > 0

def save_cr_contamination_info(frame, i3mctree_name, allowed_max_bg=0):
    cr_contamination = check_cr_contamination(frame, i3mctree_name, allowed_max_bg)
    tree = frame['I3EventHeader']
    if cr_contamination:
        return (tree.run_id, tree.sub_run_id, tree.event_id, tree.sub_event_id, tree.sub_event_stream)
    return None

def process_file(file_path):
    clean_events_csv = []
    file = dataio.I3File(file_path)
    while file.more():
        frame = file.pop_frame()
        if frame.Stop == icetray.I3Frame.Physics:
            clean_event = save_cr_contamination_info(frame, 'I3MCTree', 0)
            if clean_event:
                clean_events_csv.append(clean_event)
    return clean_events_csv

def process_directory(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.i3.zst')]
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, files)
    clean_events_csv = [event for result in results for event in result]
    return clean_events_csv

if __name__ == '__main__':
    base_dir = '/data/sim/IceCube/2020/filtered/level2/neutrino-generator/22017/'
    for x in range(4):
        sub_dir = f'{base_dir}000{x}000-000{x}999/'
        clean_events_csv = process_directory(sub_dir)
        output_file = f'/home/jnikolai/22017/clean_event_ids_000{x}000-000{x}999.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run ID', 'Subrun ID', 'Event ID', 'Subevent ID', 'Subevent Stream'])
            writer.writerows(clean_events_csv)
