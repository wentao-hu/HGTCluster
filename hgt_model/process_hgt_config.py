import json
import sys


def get_feature_config(config_name):
    cur_path = sys.path[0]
    with open(cur_path + '/' + config_name, 'r') as f:
        config = json.load(f)

    feature_config = {}
    for key in ['userFeatures', 'authorFeatures', 'noteFeatures']:
        features = config[key]
        attr_types = []
        attr_dims = []
        feature_dict = {}
        #  ["st_concept_profile", -1, -1, 32, "bucket-string", 10000],
        for feature in features:
            is_multi_val = (feature[2] == -1)
            attr_type = None
            attr_dim = None
            if len(feature) == 4:
                if is_multi_val:  # position 3 is -1 means multi-val string
                    attr_type = ('string', None, True)
                else:
                    attr_type = 'string'
                attr_dim = feature[3]
            else:
                type = feature[4]
                if type in ("int", "float"):
                    attr_type = type
                    attr_dim = None
                elif type == "bucket-int":
                    bucket = feature[5]
                    attr_dim = feature[3]
                    if is_multi_val:
                        raise Exception("bucket-int cannot be multival: {}".format(feature))
                    else:
                        attr_type = ("int", bucket)
                elif type == "bucket-string":
                    bucket = feature[5]
                    attr_dim = feature[3]
                    if is_multi_val:
                        attr_type = ("string", bucket, True)
                    else:
                        attr_type = ("string", bucket)
                else:
                    raise Exception("invalid feature config = {}".format(feature))
            assert attr_type is not None
            attr_types.append(attr_type)
            attr_dims.append(attr_dim)
            print("append feature config: {}, attr_type={}, attr_dim={}".format(feature[0], attr_type, attr_dim))
        feature_dict['attr_types'] = attr_types
        feature_dict['attr_dims'] = attr_dims
        feature_config[key] = feature_dict
    return feature_config


if __name__ == "__main__":
    feature_config = get_feature_config("hgt_config_with_bucket.json")
