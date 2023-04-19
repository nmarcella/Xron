import numpy as np
import ovito
from ovito.io import import_file
from ovito.modifiers import *
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from ovito.data import *
from scipy.integrate import trapz
from fastdist import fastdist
import pyperclip

class AnalysisPipeline:
    def __init__(self, pipeline, framerange):
        self.pipeline = pipeline
        self.framerange = framerange
        self.results = {
            "t_v_ave": {},
            "t_r_ave": {},
            "t_all_positions": {},
            "t_c_ads": {},
            "t_cn_ptpt": {},
        }

    def run(self):
        for frame in self.framerange:
            data = self.pipeline.compute(frame)
            finder = CutoffNeighborFinder(3, data)
            finder2 = CutoffNeighborFinder(3.4, data)

            ptatom_index, catom_index, oatom_index = self.get_atom_indices(data)
            all_positions = data.particles.positions[...]

            interesting_c, interesting_pt, interesting_o = self.find_interesting_atoms(
                catom_index, ptatom_index, oatom_index, finder
            )

            ptptcn, ptpt_R, ptpt_msrd = self.compute_ptpt_metrics(
                ptatom_index, finder2
            )

            c_ads = len(interesting_c)
            pt_atom_coords = all_positions[ptatom_index]

            self.update_results(frame, ptpt_msrd, ptpt_R, pt_atom_coords, c_ads, np.mean(ptptcn))

        self.process_results()

    def get_atom_indices(self, data):
        ptatom_index = np.where(data.particles['Particle Type'].array == 1)[0]
        catom_index = np.where(data.particles['Particle Type'].array == 2)[0]
        oatom_index = np.where(data.particles['Particle Type'].array == 3)[0]
        return ptatom_index, catom_index, oatom_index

    def find_interesting_atoms(self, catom_index, ptatom_index, oatom_index, finder):
        interesting_c, interesting_pt, interesting_o = [], [], []
        for i in catom_index:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                n_count = 0
                for l in neigh:
                    if l in ptatom_index:
                        n_count += 1
                        interesting_pt.append(l)
                if n_count > 0:
                    interesting_c.append(i)

        interesting_c = np.unique(np.array(interesting_c))
        interesting_pt = np.unique(np.array(interesting_pt))

        for i in interesting_c:
            neigh = [neigh.index for neigh in finder.find(i)]
            if len(neigh) > 0:
                for l in neigh:
                    if l in oatom_index:
                        interesting_o.append(l)

        interesting_o = np.unique(np.array(interesting_o))

        return interesting_c, interesting_pt, interesting_o

    def compute_ptpt_metrics(self, ptatom_index, finder2):
        ptptcn, ptpt_R, ptpt_msrd = [], [], []
        for i in ptatom_index:
            neigh = [neigh.index for neigh in finder2.find(i)]
            rs = [neigh.distance for neigh in finder2.find(i)]
            distances = [[l, rs[i]] for i, l in enumerate(neigh) if l in ptatom_index]

            distances_unique = np.unique(np.array(distances), axis=0)
            distance = np.mean(distances_unique[:, 1], axis=0)
            var = np.var(distances_unique[:, 1], axis=0)

            ptpt_msrd.append(var)
            ptptcn.append(len(distances))
            ptpt_R.append(distance)
        ptptcn = np.array(ptptcn)
        ptpt_R = np.array(ptpt_R)
        ptpt_msrd = np.array(ptpt_msrd)
        ptpt_msrd = np.mean(ptpt_msrd)
        ptpt_R = np.mean(ptpt_R)
        cn_ptpt = np.mean(ptptcn)

        return ptptcn, ptpt_R, ptpt_msrd

    def update_results(self, frame, ptpt_msrd, ptpt_R, pt_atom_coords, c_ads, cn_ptpt):
        self.results["t_v_ave"][frame] = ptpt_msrd
        self.results["t_r_ave"][frame] = ptpt_R
        self.results["t_all_positions"][frame] = pt_atom_coords
        self.results["t_c_ads"][frame] = c_ads
        self.results["t_cn_ptpt"][frame] = cn_ptpt

    def process_results(self):
        t_cns = np.array(list(self.results["t_cn_ptpt"].values()))
        t_all_msrd = np.array(list(self.results["t_v_ave"].values()))
        t_all_r = np.array(list(self.results["t_r_ave"].values()))
        t_all_c_ads = np.array(list(self.results["t_c_ads"].values()))

        self.results["t_cn_ptpt"] = t_cns
        self.results["t_v_ave"] = t_all_msrd
        self.results["t_r_ave"] = t_all_r
        self.results["t_c_ads"] = t_all_c_ads
