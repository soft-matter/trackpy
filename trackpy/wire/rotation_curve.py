import os
import numpy as np
from collections import OrderedDict
from yaml_serialize import Serializable
from pandas import Series, read_table

class RotationCurve(Serializable):
    """Container for wire rotation curve and associated meta data.
    
    Parameters
    ----------
    data : Series or filename of text file to parse
    video_number : optional
    age : like '00:04:03', optional
    trial : any string, optional
    field : in Gauss, optional
    seq_number : counting split sub-curves
    wire length : in microns, optional
    remark : any string, optional
    split_curves : list of child RotationCurve objects

    Example 
    -------
    >>> a = mr.wire.RotationCurve('filename')
    >>> mr.save('some filename', a)
    >>> b = mr.load('some_filename')
    """

    def __init__(self, data, fps=None, video_number=None, age=None,
                 trial=None, field=None, seq_number=None, wire_length=None, 
                 remark=None, split_curves=None):
        if isinstance(data, Series):
            self.data = Series.values
            if fps is None:
                raise ValueError("fps must be specified")
        if isinstance(data, str):
            if not os.path.isfile(data):
                raise ValueError("data must be a Series or a filename")
            df = read_table(data)
	    self.data = df.iloc[:, 0].values
            if fps is None:
                spf = df.iloc[:, 1].diff().min()
                assert spf != 0, "Could not infer seconds between frames."
                self.fps = np.rint(1/spf.astype(float)).astype(int)
        self.video_number = video_number
        self.age = age
        self.trial = trial
        self.field = field
        self.seq_number = seq_number
        self.wire_length = wire_length
        self.remark = remark
        self.split_curves = split_curves

    def __repr__(self):
        split_curve_count = None if self.split_curves is None else \
  len(self.split_curves)
        optional_fields = OrderedDict([('Video', self.video_number), 
                           ('Age', self.age), 
                           ('Trial', self.trial),
                           ('Field (G)', self.field), 
                           ('Seq. Number', self.seq_number),
                           ('Wire Length (microns)', self.wire_length), 
                           ('', self.remark)])
        output = ""
        for k, v in optional_fields.items():
            if v is not None:
                if len(k) > 0:
                    output += "{}: ".format(k)
                output += "{}\n".format(v)
        return output[:-1]  # drop the final \n

    def add_child(self, child):
        self.split_curves.append(child)
