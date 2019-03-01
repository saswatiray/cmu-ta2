"""
A module for loading primitives.
"""

import d3m.index
import d3m.metadata.base as metadata
import logging
import primitivedescription

logging.basicConfig(level=logging.INFO)

def list_primitives():
    """
    Returns a list of all primitives, as PrimitiveMetadata[].
    """
    primcs = []
    prims = d3m.index.search()
        
    for pc in prims:
        # Do not load expensive primitives! These ones have been reported to take very long making TA2 unavailable!
        if 'cornell' in pc or 'umich' in pc:
            continue
        try:
            primitive_obj = d3m.index.get_primitive(pc)
        except:
            continue

        if hasattr(primitive_obj, 'metadata'):
            try:
                primitive = PrimitiveMetadata(primitive_obj.metadata, primitive_obj)
            except:
                continue
            primcs.append(primitive)

    return primcs
 
class PrimitiveMetadata(object):
    """
    A class that mainly just contains primitive metadata.
    """
    def __init__(self, metadata, classname):
        self.id = metadata.query()['id']
        self.name = metadata.query()['name']
        self.version = metadata.query()['version']
        self.python_path = metadata.query()['python_path'] 
        self.classname = classname
        self.family = metadata.query()['primitive_family']
        self.digest = metadata.query()['digest']

        self.arguments=[]
        args = metadata.query()['primitive_code']['arguments']
        for name,value in args.items():
            if value['kind'] == "PIPELINE":
                self.arguments.append(name)

        self.produce_methods=[]
        args = metadata.query()['primitive_code']['instance_methods']
        for name,value in args.items():
            if value['kind'] == "PRODUCE":
                self.produce_methods.append(name)

def load_primitives():
    """
    Return dictionary of primitivename-to-PrimitiveDescription class object.
    """
    from timeit import default_timer as timer

    start = timer()
    primitives = {}
    for p in list_primitives():

        # Do not add primitives which do not work or do not give good performance
        if 'convolutional_neural_net' in p.python_path or 'feed_forward_neural_net' in p.python_path or 'regression.k_neighbors' in p.python_path or 'loss.TorchCommon' in p.python_path:
            continue
        if 'quadratic_discriminant_analysis.SKlearn' in p.python_path or 'd3m.primitives.classification.random_forest.DataFrameCommon' in p.python_path or 'bayesian_logistic_regression.Common' in p.python_path:
            continue
        if 'decision_tree' in p.python_path or 'bernoulli_naive_bayes' in p.python_path or 'gaussian_naive_bayes' in p.python_path or 'dummy' in p.python_path or 'tensor_machines' in p.python_path or 'fast_lad' in p.python_path:
            continue
        if 'rfm_precondition' in p.python_path or 'BayesianInfRPI' in p.python_path:
            continue

        primitives[p.classname] = primitivedescription.PrimitiveDescription(p.classname, p)

    end = timer()
    logging.info("Time taken to load primitives: %s seconds", end - start)
    return primitives

