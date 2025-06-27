#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# import sys
# sys.path.append("/exps/example/yolox_voc/yolox_voc_s.py")
import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp

#def get_exp_by_name(exp_name):

    #exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    #module_name = ".".join(["yolox", "exp", "default", exp])
    #exp_object = importlib.import_module(module_name).Exp()
    #return exp_object
def get_exp_by_name(exp_name):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    """获取yolox的根目录"""
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py", }
    filename = filedict[exp_name]
    """模型名称对应的py文件"""
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    """路径拼接：根目录/exps/default/"""
    return get_exp_by_file(exp_path)


#def get_exp(exp_file=None, exp_name=None):
def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
