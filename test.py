"""
test accuracy

sample tables from chosen table , and then test every tables' accuracy.

Created on May 24, 2023

@author: Dong Hao (3396363108@qq.com)
"""

import collaborative_filtering_for_movies as cf
import numpy as np
import matplotlib.pyplot as plt

def test_accuracy(number, train_ratio_of_general_table, sim_threshold, general_table):
    """
    sample tables from general table in database and compute every table(or sample)'s accuracy

    :param number: number of tests
    :param train_ratio_of_general_table: ratio of train set in general table
    :param sim_threshold: similarity threshold
    :param general_table: parent table
    :return: score_list containing average accuracy of each train set
    """

    # store created tables
    table_list = []
    # store average accuracy of every table
    score_list = []
    # create sample tables in database according to param number, and add it to table list to facilitate delete finally
    print("create sample tables...")
    for no in range(number):
        sql_create_sample_table = f'create table test{no + 1}(' \
                                  f'select * from {general_table} where rand() < {train_ratio_of_general_table} );'
        cf.connect_to_db('root', 'root', 'movies', sql_create_sample_table, 3306, 'localhost', 'utf8')
        table_list.append(f'test{no + 1}')
    print('sample successfully :')
    print(table_list)
    # test accuracy of every sample
    for no in range(number):
        print(f'test sample {no + 1}...')
        # get every user id of every table
        sql_select_user_id_of_one_table = f'select distinct (userId + 0) from test{no + 1};'
        user_2_dimension_tuple = cf.connect_to_db('root', 'root', 'movies', sql_select_user_id_of_one_table, 3306,
                                                  'localhost', 'utf8')
        # reduce dimension to one
        user_id_list = np.array(user_2_dimension_tuple, int).flatten('F')
        # compute similarity
        sim_path = cf.compute_similarity_using_for_user('root', 'root', 3306, 'movies', f'test{no + 1}',
                                                        'localhost', 'utf8', 1.0, 'adjusted_cos')
        # find all movie ids of table
        sql_select_movie_id_of_one_table = f'select distinct (movieId + 0) from test{no + 1};'
        movie_2_dimension_tuple = cf.connect_to_db('root', 'root', 'movies', sql_select_movie_id_of_one_table, 3306,
                                                   'localhost', 'utf8')
        # reduce dimension to one
        movie_id_list = np.array(movie_2_dimension_tuple, int).flatten('F')
        # table_accuracy is to store every user's accuracy
        table_accuracy = []
        for user_id in user_id_list:
            # travel every user id of every table, and recommend items for whom
            recommendation_list = cf.recommend(user_id, '1995-01-09 00:00:00', '2019-11-21 00:00:00',
                                               sim_threshold,
                                               sim_path, f'test{no + 1}')
            # remove movie_ids not exist in table from recommendation_list
            recommendation_list = list(set(recommendation_list) & set(movie_id_list))
            # --------------------------------------delete
            # print(recommendation_list)
            # --------------------------------------delete
            # find real user's behaving items from general_table, which is the parent table of every sample table
            sql_select_real_movie_id_of_user_id = f'select (movieId + 0) from {general_table} where (userId + 0) =' \
                                                  f'{user_id}'
            real_movie_list = cf.connect_to_db('root', 'root', 'movies', sql_select_real_movie_id_of_user_id, 3306,
                                               'localhost', 'utf8')
            real_movie_list = np.array(real_movie_list, int).flatten('F')
            # --------------------------------------delete
            # print(real_movie_list)
            # --------------------------------------delete
            # if length of recommend_list is zero, that means making no recommendations.So the accuracy(for one user)
            # is 0.If else accuracy is counted as recommendations behaved by user / all recommendations
            if len(recommendation_list) == 0:
                # table_accuracy.append(0)
                # print this user's accuracy
                # print(f'table {no + 1}--{user_id} : {table_accuracy[-1]}')
                pass
            else:
                score = 1.0 * (len(set(recommendation_list)) - len(set(recommendation_list).difference(
                    set(real_movie_list)))) / len(set(recommendation_list))
                table_accuracy.append(score)
                # print this user's accuracy
                # print(f'table {no + 1}--{user_id} : {table_accuracy[-1]}')
        # add every table's average accuracy to score_list
        mean_accuracy = np.mean(table_accuracy)
        # print(f'sample (or table) {no + 1} accuracy:')
        # print(mean_accuracy)
        score_list.append(mean_accuracy)
    # delete sample tables in database
    print('delete samples ...')
    for table in table_list:
        sql_delete_test_tables = f'drop table {table}'
        cf.connect_to_db('root', 'root', 'movies', sql_delete_test_tables, 3306, 'localhost', 'utf8')
    # return every sample(or table)'s average accuracy
    print('samples have been deleted successfully')
    return score_list


if __name__ == '__main__':
    x = []
    y = []
    for i in np.arange(0.1, 0.8, 0.1):
        print(i)
        x.append(i)
        accuracy_list = test_accuracy(1, 0.5, i, 'new')
        print(accuracy_list)
        y.append(accuracy_list[0])
    plt.plot(x, y)
    plt.xlabel('similarity threshold')
    plt.ylabel('precision')
    plt.title('precision with similarity threshold')
    plt.show()


    """
    [0.019774229346592266]   
    [0.016803482206606334]
    
    [0.12919627937442965]  -- 0.3
    """
