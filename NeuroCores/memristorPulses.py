import numpy as np

from arc1pyqt.VirtualArC.parametric_device import ParametricDevice as Device


def _append_debug(message):
    """Best-effort debug logging without hard failing on missing local paths."""
    try:
        with open("debug_log.txt", "a") as debug_file:
            debug_file.write(message)
    except OSError:
        # Debug logging must never break simulation/runtime.
        pass


class memristorPulses:
    def __init__(self, dt, Ap, An, a0p, a1p, a0n, a1n, tp, tn, R): # initialise a device with parameters
        self.dt = dt
        self.Ap = Ap
        self.An = An
        self.a0p = a0p
        self.a1p = a1p
        self.a0n = a0n
        self.a1n = a1n
        self.tp = tp
        self.tn = tn
        self.R = R

    def ResistancePredict(self, pulseList):  # return the list of R for different pulses
        memristor = Device(self.Ap, self.An, self.a0p, self.a1p, self.a0n, self.a1n, self.tp, self.tn)
        res = []
        for i in pulseList:
            memristor.initialise(self.R)
            _append_debug('pulsechoice')
            line = ' '.join(str(x) for x in i)
            _append_debug(line + ', ')
            for timestep in range(int(i[1]/self.dt)):
                memristor.step_dt(i[0], self.dt)
                #print('new R: %f, old R: %f, mag: %f, pw: %f' % (self.R, memristor.Rmem, i[0], i[1]))
            _append_debug('res:%f\n' % memristor.Rmem)
            res.append(memristor.Rmem)
        del memristor
        print('pulseList:', pulseList)
        print('res:', res)
        return res

    def BestPulseChoice(self, R_expect, pulseList):  # return the pulse that can make the device reach the expected resistance
        res = self.ResistancePredict(pulseList)
        res_dist = np.absolute(np.array(res) - R_expect)
        print('res dist:', res_dist)
        _append_debug('res dist:')
        line = ', '.join(str(i) for i in list(res_dist))
        _append_debug(line + '\n')
        res_index = np.argmin(res_dist)

        return pulseList[res_index]
