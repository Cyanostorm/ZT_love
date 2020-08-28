
bool RandomTree::TrainStructureofTree(std::vector<Feature3DInfo> &featureInfos, string filename) {
    mNumberofClassesforTraining = featureInfos.size();

    if (!CalalutionPreData(featureInfos)) {
        return false;
    }

    vector<int> *leave_index_of_descriptors = new vector<int>[mNumberofClassesforTraining];
    vector<int> *leave_index_of_descriptors_old = new vector<int>[mNumberofClassesforTraining];

    vector<bool> bselected(10000, false);
    for (int tree_idx = 0; tree_idx < mNumberofTrees; tree_idx++) { // mNumberofTrees 树的个数(6)
        cout << "tree_idx: " << tree_idx << endl;
        if (tree_idx > 0) { // 第 0 棵树,不执行
            for (int tree_idx_trained = 0; tree_idx_trained < tree_idx; ++tree_idx_trained) {
                for (int level = 0; level < 3; level++) {
                    for (int j = 0; j < (1 << level); j++) {
                        int test_idx_trained = mSelectedTestsPerLevel[tree_idx_trained * mNumberofLevelsPerTree + level][j].test_idx;
                        bselected[test_idx_trained] = true;
                    }
                }
            }

            for (int col = 0; col < DESCRIPTOR_DIM; ++col) {
                for (int tree_idx_trained = 0; tree_idx_trained < tree_idx; ++tree_idx_trained) {
                    for (int level = 0; level < 3; level++) {
                        for (int j = 0; j < (1 << level); j++) {
                            int row = mSelectedTestsPerLevel[tree_idx_trained * mNumberofLevelsPerTree + level][j].test_idx;
                            if (mCorrelation[row * DESCRIPTOR_DIM + col] > mMaxCorrelation[col]) {
                                mMaxCorrelation[col] = mCorrelation[row * DESCRIPTOR_DIM + col];
                            }
                        }
                    }
                }
            }

            vector<pair<float, int>> correlationIdx;
            for (int col = 0; col < DESCRIPTOR_DIM; ++col) {
                correlationIdx.push_back(pair<float, int>(mMaxCorrelation[col], col));
            }

            std::stable_sort(correlationIdx.begin(), correlationIdx.end(), CmpCorrelationIdxInverse);
            for (int test_idx = 0; test_idx < 2 * tree_idx * 7; ++test_idx) {
                bselected[correlationIdx[test_idx].second] = true;
            }
        }

        vector<pair<int, float>> *inverted_file = NULL;
        vector<pair<int, float>> *inverted_file_previous_level = NULL;
        inverted_file_previous_level = new vector<pair<int, float>>[1];
        
        for (int class_idx = 0; class_idx < mNumberofClassesforTraining; class_idx++) {
            vector<bitset<DESCRIPTOR_DIM>> &elem_desc = featureInfos[class_idx].descriptorsBin;
            vector<int> &elem_leave_index = leave_index_of_descriptors[class_idx];
            vector<int> &elem_leave_index_old = leave_index_of_descriptors_old[class_idx];

            // TODO：总长度为3D点数目，每个vector长度为该3D点描述子的个数
            elem_leave_index.resize(featureInfos[class_idx].view_list.size());
            elem_leave_index_old.resize(featureInfos[class_idx].view_list.size());

            if (elem_desc.size() >= TRACK_LENGTH) {
                inverted_file_previous_level[0].push_back(pair<int, float>(class_idx, float(elem_desc.size()))); // 三维点编号,和对应三维点的描述子个数
                for (int p = 0; p < elem_leave_index.size(); p++) {
                    elem_leave_index[p] = 0;
                    elem_leave_index_old[p] = 0;
                }
            }
        }

        for (int level = 0; level < mNumberofLevelsPerTree; level++) {
            cout << "level: " << level << endl;
            //number of test to select
            int test_to_select = (1 << level); // 每一层的节点个数
            //number of nodes produced when applying these newly selected tests
            int number_of_nodes = (2 << level); // 所有的节点(叶子节点+非叶子结点) + 1
            if (inverted_file != NULL) {
                delete[] inverted_file;
                inverted_file = NULL;
            }
            inverted_file = new vector<pair<int, float>>[number_of_nodes]; // 所有的节点(叶子节点+非叶子结点) + 1

            for (int j = 0; j < test_to_select; j++) { // test_to_select 当前层的非叶子节点数
                // cout << "test_to_select: " << j << endl;

                vector<pair<int, float>> inverted_file_per_nodes[2];
                vector<pair<int, float>> &elem_ivt_pre = inverted_file_previous_level[j]; //the set of the fp used to chose the jth test
                // cout << "the node " << j << "th size is: " << elem_ivt_pre.size() << endl;

                if (elem_ivt_pre.size() == 0) {
                    mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].test_idx = rand() % DESCRIPTOR_DIM; //-1;
                    mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].threshold = 0;
                    continue;
                }

                if (elem_ivt_pre.size() == 1) {                                                                                                                                     //can not be divided any more
                    mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].test_idx = rand() % DESCRIPTOR_DIM; //-1;
                    mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].threshold = 0;
                    inverted_file[j << 1] = elem_ivt_pre; // 这里j<<1表示在所有的节点中的序号，根节点为0，多一个节点的空间，这里假设j=2，<<1后为4，即对应位置
                    continue;
                }

                cout << "select test "
                     << "of the" << j << "th node of level " << level << " for the " << tree_idx << " th tree" << endl
                     << flush;
                float min_entropy, max_difference;
                min_entropy = 1e20;
                max_difference = -1e20;
                int min_test_idx;
                int min_threshold_idx;
                float opt_mean_error;
                float opt_mean;

                int selected_test_num;
                if (level == 0) {
                    selected_test_num = 100; // 测试100次，即从DESCRIPTOR_DIM维中随机取100次，从这100次中选择1维作为该节点的分类依据
                } else {
                    selected_test_num = 100;
                }
                srand(int(time(0)));
                #pragma omp parallel for schedule(dynamic)
                for (int test = 0; test < selected_test_num; test++) { // selected_test_num 做100次测试
                    int test_idx;
                    do {
                        test_idx = rand() % DESCRIPTOR_DIM;
                    } while (((level < 3) && bselected[test_idx]) || mBlocked[test_idx]); // 前三层不选重复的维数

                    float min_entropy_thr, entropy_thr;
                    min_entropy_thr = 1e20;
                    int min_threshold_idx_thr;
                    float opt_mean_error_thr;
                    float opt_mean_thr;

                    for (int threshold_idx = 1; threshold_idx < 2; threshold_idx++) { //threshold_idx 只能为1, 不会循环
                        inverted_file_per_nodes[0].clear();
                        inverted_file_per_nodes[1].clear(); // vector<pair<int,float> > inverted_file_per_nodes[2];
                        float mean_error = 0.0;
                        float mean = 0.0;
                        int class_count_t = 0;
                        int total_query_feature = 0;
                        int survived_query_feature = 0;
                        for (int p = 0; p < elem_ivt_pre.size(); p++) { // elem_ivt_pre: the set of the fp used to chose the jth test  第0层，elem_ivt_pre.size() 为3D点总数
                            if (p % 10000 == 0) {
                                std::cout << "test: " << test << " pre index: " << p << std::endl;
                            }

                            int fp_id = elem_ivt_pre[p].first; // pair<int,float>  三维点编号,和对应三维点的描述子个数
                            // cout << "fp_id: " << fp_id << endl;
                            vector<bitset<DESCRIPTOR_DIM>> &elem_desc = featureInfos[fp_id].descriptorsBin;
                            vector<int> &elem_leave_index_old = leave_index_of_descriptors_old[fp_id];

                            if (elem_desc.size() < TRACK_LENGTH) {
                                continue;
                            }

                            int sample_count_per_fp[2] = {0, 0};
                            for (int sample_idx = 0; sample_idx < elem_desc.size(); sample_idx++) {
                                bitset<DESCRIPTOR_DIM> &desc_temp = elem_desc[sample_idx];
                                int leave = desc_temp[test_idx] ? 1 : 0; // test_idx 代表随机取的描述子的某一维
                                sample_count_per_fp[leave]++; // 统计 0 或 1 的个数 ?
                            }

                            for (int sample_idx = 0; sample_idx < elem_desc.size(); sample_idx++) { //elem_desc.size() 描述子个数
                                if (elem_leave_index_old[sample_idx] != j) { // j 当前层的非叶子结点编号 elem_leave_index_old[sample_idx] 里面全是零??
                                    //cout << "elem_leave_index_old[sample_idx]" << elem_leave_index_old[sample_idx] << endl;
                                    continue;
                                } //这个地方有明显的bug? 只有当j = 0的时候才会work ??? 没有bug, 后面有更新 elem_leave_index_old[sample_idx] !!!

                                // if (p % 1000 == 0) {
                                //     cout << "j=" << j << endl;
                                //     cout << "elem_leave_index_old[sample_idx]=" << elem_leave_index_old[sample_idx] << endl;
                                // }

                                bitset<DESCRIPTOR_DIM> &desc_temp = elem_desc[sample_idx];
                                int leave_fp = desc_temp[test_idx] ? 1 : 0;
                                survived_query_feature += sample_count_per_fp[leave_fp] - 1; // survived_query_feature 所有三维点的所有描述子查询通过次数之和 假如 这个三维点共有5个描述子,其中的某一位 3个1, 两个 0, 则通过次数为(1+1+2+2+2 = 8)
                                total_query_feature += sample_count_per_fp[0] + sample_count_per_fp[1] - 1; //假如每个三维点有 5 个描述子, 则  total_query_feature = (5-1)*5*三维点总数
                                class_count_t++;
                                if (leave_fp == 0) {
                                    mean += 1.0; //统计所有三维点的所有描述子的 test_idx 为 0 的个数 ???
                                }
                            }
                            //cout << "elem_ivt_pre.size()=" << elem_ivt_pre.size() << endl;
                            //cout << "jjjjjjjjjjjjjjjjjjjjjjjjj=" << j << endl;
                            /*
                            survived_query_feature+=sample_count_per_fp[leave_fp];
                            total_query_feature+=sample_count_per_fp[0]+sample_count_per_fp[1];
                            class_count_t++;
                            if(leave_fp==0){
                                mean+=1.0;
                            }*/
                        } //p

                        if (total_query_feature != 0) {
                            mean_error = 1.0 - float(survived_query_feature) / float(total_query_feature);
                        } else {
                            mean_error = 0.0;
                        }
                        mean /= class_count_t;

                        if (fabs(mean - 0.5) == 0.5) { // 全是零或者全是1, 均匀性. 因为这是所有的三维点的描述子,不同三维点的描述子之间并不是匹配的,所以与错误率无关
                            entropy_thr = 1e20 + 1.0;
                        } else {
                            if (level < 17) {
                                entropy_thr = 8.0 * mean_error + 1.0 / (0.5 - fabs(mean - 0.5));
                            } else {
                                entropy_thr = 8.0 * mean_error + 1.0 / (0.5 - fabs(mean - 0.5));
                            }
                        }
                        /*
                        entropy_thr=0.0;
                        float total_sample_count_this_node=0;
                        for(int leave_idx=0;leave_idx<2;leave_idx++){
                            if(inverted_file_per_nodes[leave_idx].size()==0){
                                continue;
                            }
                            float temp_sample_count=0.0;
                            float temp_entropy=0.0;

                            for(int p=0;p<inverted_file_per_nodes[leave_idx].size();p++){
                                int fp_idx=inverted_file_per_nodes[leave_idx][p].first;
                                temp_sample_count+=inverted_file_per_nodes[leave_idx][p].second/float(sample_num_per_kp[fp_idx]);
                            }
                            for(int p=0;p<inverted_file_per_nodes[leave_idx].size();p++){
                                int fp_idx=inverted_file_per_nodes[leave_idx][p].first;
                                float probability=(inverted_file_per_nodes[leave_idx][p].second/float(sample_num_per_kp[fp_idx]))/temp_sample_count;
                                temp_entropy+=-probability*log(probability);
                            }
                            entropy_thr+=temp_entropy*temp_sample_count;
                            total_sample_count_this_node+=temp_sample_count;
                        }
                        entropy_thr/=total_sample_count_this_node;
                        entropy_thr/=log(2.0);
                        */
                        if (entropy_thr < min_entropy_thr) { // min_entropy_thr=1e20;
                            min_entropy_thr = entropy_thr;
                            min_threshold_idx_thr = threshold_idx; // threshold_idx = 1
                            opt_mean_error_thr = mean_error;
                            opt_mean_thr = mean;
                        }
                    } //threshold_idx

                    #pragma omp critical
                    {
                        if (min_entropy_thr < min_entropy) {
                            min_entropy = min_entropy_thr; // min_entropy=1e20;
                            min_test_idx = test_idx;
                            min_threshold_idx = min_threshold_idx_thr;
                            opt_mean_error = opt_mean_error_thr;
                            opt_mean = opt_mean_thr;
                        }
                    }
                } //test

                mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].test_idx = min_test_idx; //这是训练的结果
                mSelectedTestsPerLevel[tree_idx * mNumberofLevelsPerTree + level][j].threshold = -20 + min_threshold_idx * 20; // -20+min_threshold_idx*20 = 0 ???

                cout<<"opt error and mean is: "<<opt_mean_error<<" "<<opt_mean<<endl;
                cout<<"min_test_idx: "<< min_test_idx <<endl;

                if (level == 0) {
                    bselected[min_test_idx] = true;
                }

                // insert the samples in the jth node in the new inverted file and update the indices of these samples;
                for (int p = 0; p < elem_ivt_pre.size(); p++) { //elem_ivt_pre: the set of the fp used to chose the jth test  elem_ivt_pre.size() = 523319
                    // if (p % 10000 == 0) {
                    //     std::cout << "将节点按照min_test_idx位分到左子节点和右子节点, index: " << p << std::endl;
                    // }
                    // std::cout << "将节点按照min_test_idx位分到左子节点和右子节点：第i个：" << p << endl;
                    int fp_id = elem_ivt_pre[p].first;
                    vector<bitset<DESCRIPTOR_DIM>> &elem_desc = featureInfos[fp_id].descriptorsBin;
                    vector<int> &elem_leave_index = leave_index_of_descriptors[fp_id];
                    vector<int> &elem_leave_index_old = leave_index_of_descriptors_old[fp_id];
                    //assert(elem_leave_index_old[0]==j);
                    //bitset<CANDIDATE_TEST_COUNT>& desc_temp=elem_desc[0];
                    //int leave=(j<<1)+(desc_temp[min_test_idx]?1:0);
                    //inverted_file[leave].push_back(pair<int,float>(fp_id,1.0));
                    //elem_leave_index[0]=leave;

                    if (elem_desc.size() < TRACK_LENGTH) {
                        continue;
                    }

                    for (int sample_idx = 0; sample_idx < elem_desc.size(); sample_idx++) { //一个三维点对应多少个描述子
                        if (elem_leave_index_old[sample_idx] != j) {
                            continue;
                        }

                        bitset<DESCRIPTOR_DIM> &desc_temp = elem_desc[sample_idx];
                        // for (size_t index=0; index < desc_temp.size();index++) {
                        //     cout << desc_temp[index] << " ";
                        // }
                        // cout << endl;
                        int leave_temp = (j << 1) + (desc_temp[min_test_idx] ? 1 : 0); // j左移1位 j*2  leave_temp 这个变量什么意思 ???  j 表示当前层的非叶子结点
                        bool bnewclass = true;
                        if (inverted_file[leave_temp].size() > 0) {
                            int q = inverted_file[leave_temp].size() - 1;
                            if (inverted_file[leave_temp][q].first == fp_id) {
                                inverted_file[leave_temp][q].second += 1.0;
                                bnewclass = false;
                            }
                        }
                        // std::cout << "leave_temp: " << leave_temp << endl;
                        if (bnewclass) {
                            inverted_file[leave_temp].push_back(pair<int, float>(fp_id, 1.0)); //inverted_file 有 (2<<level) 个 vector
                            // 把所有的三维点按照之前所有层叶子节点内的测试分类, 放到 inverted_file[leave_temp] 这个文件中.
                        }

                        elem_leave_index[sample_idx] = leave_temp;
                        // 这个变量, 标记着每个三维点有多少个描述子落在了inverted_file[leave_temp] 中, 并没有指明具体哪一个描述子落在inverted_file[leave_temp]中. 而 指明了其对应的描述子在哪一个 leave_temp 中.
                    }
                }
            } //j

            //when all the tests in the levelth level have been selected copy inverted_file int inverted_file_pervious_level
            for (int class_idx = 0; class_idx < mNumberofClassesforTraining; class_idx++) {
                leave_index_of_descriptors_old[class_idx] = leave_index_of_descriptors[class_idx];
            }

            if (inverted_file_previous_level != NULL) {
                delete[] inverted_file_previous_level;
                inverted_file_previous_level = NULL;
            }

            inverted_file_previous_level = new vector<pair<int, float>>[2 << level];
            for (int j = 0; j < (2 << level); j++) {
                inverted_file_previous_level[j] = inverted_file[j];
            }
        } //level

        if (inverted_file != NULL) {
            delete[] inverted_file;
            inverted_file = NULL;
        }
        if (inverted_file_previous_level) {
            delete[] inverted_file_previous_level;
            inverted_file_previous_level = NULL;
        }
    } //tree_idx;

    ofstream file_tree_mp(filename.c_str(), ios::binary);
    if (!file_tree_mp.is_open()) {
        cout << "save fern_mp file failed" << endl;
    } else {
        SaveMp(file_tree_mp);
    }
    file_tree_mp.close();

    for (int i = 0; i < mNumberofTrees; ++i) {
        for (int j = 0; j < mNumberofLevelsPerTree; ++j) {
            for (int k = 0; k < (1 << j); ++k) {
                int test_idx = mSelectedTestsPerLevel[i * mNumberofLevelsPerTree + j][k].test_idx; // 第i棵树 的 第j层 的 第k个节点
                mSimpliedTests[i * mNumberofLeavesPerTree + (1 << j) - 1 + k] = test_idx; // 和 mSelectedTestsPerLevel 中的 test_idx 数据应该是一样的, 只是数据结构不一样
            }
        }
    }

    return true;
}








void RandomTree::TrainforDatabase(std::vector<Feature3DInfo> &featureInfos, string filename) {

    for (size_t i = 0; i < featureInfos.size(); i++) {
        for (size_t j = 0; j < featureInfos[i].descriptorsBin.size(); j++) {

            bitset<DESCRIPTOR_DIM> &desc = featureInfos[i].descriptorsBin[j];
            vector<int> leavesIndex = DropTreeMp(desc);
            for (auto a : leavesIndex) {
                cout << a << " ";
            }
            cout << endl;
            // cout << "feature: " << i << " view: " << j << " valaid num: " << leavesIndex.size() <<endl;

            if (leavesIndex.size() > 0) {
                for (int k = 0; k < mNumberofTrees; k++) {
                    vector<FP> &elemIvtFp = mInvertedFileFp[k * mNumberofLeavesPerTree + leavesIndex[k]]; //第 k 棵树的第 leaves_index[k] 个叶子节点对应的 vector
                    bool bnewpoint = true;
                    // if (elemIvtFp.size() > 0) {
                    //     if (elemIvtFp[elemIvtFp.size() - 1].mDescIdx == i) {
                    //         bnewpoint = false;
                    //         break;
                    //     }
                    // }

                    if (bnewpoint) {
                        elemIvtFp.push_back(FP(i, j)); // 把二维特征点的标号放入相应的叶子节点容器里
                    }
                }
            }
        }
    }

    FinalizeTraining(filename);
}
