import pandas as pd


def groupby_and_sort():
    """Groupby and sort elements of each group
    """
    class_colname = 'model_name'
    data = {'model_name': ['vae1', 'vae1', 'vae1', 
                            'gan1', 'gan1',
                            'real'],
            'img_fps': ['vae-fp3.png', 'vae-fp2.png','vae-fp1.png',
                        'gan-fp2.png', 'gan-fp1.png', 
                        'real-fp.png']
            }

    df = pd.DataFrame.from_dict(data)
    print('df:\n', df)

    #inplace sorting
    # df.sort_values(
    #     by=[class_colname],
    #     axis=0,
    #     ascending=True,
    #     inplace=False,
    #     ignore_index=True
    # )

    df_grouped = df.groupby(class_colname)
    df_sorted = df_grouped.apply(lambda x: x.sort_values(by='img_fps', 
                                                        ascending=True,
                                                        ignore_index=True)
                                )
    return df_sorted
                  
                  
def make_new_df_from_subset_of_each_group():
    """Given a df, take each df_g, groupby class_colname and
    create a new df consisting max_n number of rows from each group's data (df_g)"""
    class_colname = 'model_name'
    
    # Create a test dataframe
    _data = {'model_name': ['vae1', 'vae1', 'vae1', 
                            'gan1', 'gan1',
                            'real'],
            'img_fps': ['vae-fp3.png', 'vae-fp2.png','vae-fp1.png',
                        'gan-fp2.png', 'gan-fp1.png', 
                        'real-fp.png']
            }

    df = pd.DataFrame.from_dict(_data)
    # print('df:\n', _df)
    
    # Create a sub dataframe with max_n number of rows from each group
    max_n_per_class = 1
    subgroups = [ ] 
    for g_name, df_g in df.groupby(class_colname):
        print(g_name, len(df_g))
        df_g = df_g.sort_values(by='img_fps', ascending=True, inplace=False, ignore_index=True)
        print(df_g)
    #     print(df_g[:1])
    #     break
        subgroups.append(df_g[:max_n_per_class])
    subdf = pd.concat(subgroups)
    return subdf
    
