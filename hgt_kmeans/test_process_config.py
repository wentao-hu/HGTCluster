from process_hgt_config import get_feature_config


feature_config = get_feature_config('hgt_config_simple.json')
for key,val in feature_config.items():
    print('\n\nfeature type:',key)
    print('\nattr_types',len(val['attr_types']),val['attr_types'])
    print('\nattr_dims',len(val['attr_dims']),val['attr_dims'])
    # count=0
    # # for attr_type,attr_dim in zip(val['attr_types'],val['attr_dims']):
    # #     print("feature",count,"attr_type",attr_type,"attr_dim",attr_dim)
    # #     count+=1
    # attr_types = val['attr_types']
    # attr_dims = val['attr_dims']
    # input_dim = sum([1 if not i else i for i in attr_dims])
    # print('input_dim',input_dim)
