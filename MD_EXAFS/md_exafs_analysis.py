# Read dat from master directory
from basic.io import *
from basic.ftt_larch import *
from basic.pre_process import *

class EXAFS_Chi:
    def __init__(self, all_xmu, n_cut):
        self.output_files_dict = {}
        self.good_frames = []
        self.xmu = []
        self.chi = []
        self.mean_chi = []
        self.ft_data = []
        
        for i in all_xmu:
            frame, absorber = np.array(i.split('\\'))[[-3,-2]]
            if frame not in self.output_files_dict.keys():
                self.output_files_dict[frame] = {}
            self.output_files_dict[frame][absorber] = i

        for i in self.output_files_dict.keys():
            if len(self.output_files_dict[i]) > n_cut:
                self.good_frames.append(i)

        for i in self.good_frames:
            for j in self.output_files_dict[i].keys():
                self.xmu.append(self.output_files_dict[i][j])

        for i in self.xmu:
            raw=[l for l in read_lines(i) if l.split()[0]!='#']
            number_form=np.asarray([[float(s) for s in l.split()] for l in raw])[:,[2,5]]
            self.chi.append(number_form)

        self.chi = np.asarray(self.chi)
        self.mean_chi = np.mean(np.array(self.chi), axis=0)
        self.mean_chi_i =  intpol(k2(np.mean(np.array(self.chi), axis=0)), kspace)

        kstep=0.05
        nfft=2048
        rmax_out=10
        rstep = pi/(kstep*nfft)
        irmax = int(min(nfft/2, 1.01 + rmax_out/rstep))
        r= rstep * arange(irmax)
        chi_data, window = xftf_prep(self.mean_chi.transpose()[0], self.mean_chi.transpose()[1], kmin=3, kmax=16, kweight=2, window='kaiser')

        fftdata=xftf_fast(chi_data*window)

        x = r[:irmax]
        y = sqrt(fftdata.real**2 + fftdata.imag**2)[:irmax]
        self.ft_data = [x,y]