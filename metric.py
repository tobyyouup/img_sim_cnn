import re
import datetime
import sys
import math
import numpy as np

#s_score = 'score'
s_label = 'label'
s_price = 'price'

class auc_img():
    def __init__(self, k=999999):
        self.k = k

    def __call__(self, data):
        data = sorted(data, key=lambda x:x[1], reverse=True)   
        
        auc = 0.0
        k0 = 0
        k1 = 0
        tp0 = 0.0
        fp0 = 0.0
        tp1 = 1.0
        fp1 = 1.0
        P = 0
        N = 0
        ap = 0.0
        k0 = 0
        k1 = 0
        pos = 0

        for (act, val) in data:
            if act>1e-9:
                P += 1.0
            else:
                N += 1.0

        if P==0:
            return 0.0
        if N==0:
            return 1.0

        for (act, val) in data:
            if act > 1e-9:
                k1 += 1
                tp1 = float(k1)/P
                fp1 = float(k0)/N
                auc += (fp1-fp0)*(tp1+tp0)/2
                #print("kk", (fp1-fp0)*(tp1+tp0)/2, fp1, fp0, tp1, tp0, k1, k0)
                tp0 = tp1
                fp0 = fp1
            else:
                k0 += 1
                
        auc += 1.0 - fp1
        return auc


class metric_base:
    def __init__(self):
        pass

    def __call__(self, df):
        pass

    def check_feature_meet(self, df):
        return self.get_features_required() < set(df.columns)

class mrr(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        mrr = 0.0
        k = 0
        for idx, (name,row) in enumerate(df[:self.k].iterrows()):
            score = row[self.s_score]
            label = row[s_label]
            if label == 1:
                mrr += 1.0 / (idx + 1)
                k += 1
        if k > 0:
            mrr /= k
        return mrr


class mrr_weighted(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score
    
        return set([s_score, s_label, s_price])

    def __call__(self, df):
        mrr = 0.0
        k = 0
        for idx, (name,row) in enumerate(df[:self.k].iterrows()):        
            score = row[self.s_score]
            label = row[s_label]
            price = row[s_price]
            if label > 0:
                mrr += 1.0 / (idx + 1) * price 
                k += 1
        if k > 0:
            mrr /= k
        return mrr

class ap(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        ap = 0.0
        k0 = 0
        k1 = 0
        for name, row in df[:self.k].iterrows():                        
            score = row[self.s_score]
            label = row[s_label]
            if label == 1:
                k1 += 1
                ap += float(k1)/float(k0+k1)
            else:
                k0 += 1
        if k1 > 0:
            ap /= k1
        return ap

class ap_weighted(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        ap = 0.0
        k0 = 0
        k1 = 0
        pos = 0

        for name, row in df[:self.k].iterrows():
            score = row[self.s_score]
            label = row[s_label]
            price = row[s_price]
            if label > 0:
                k1 += price 
                pos += 1
                ap += float(k1)/float(k0+k1)
            else:
                k0 += 1
        if pos>0:
            ap /= pos
        return ap

class auc(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        auc = 0.0
        k0 = 0
        k1 = 0
        tp0 = 0.0
        fp0 = 0.0
        tp1 = 1.0
        fp1 = 1.0
        P = 0
        N = 0
        ap = 0.0
        k0 = 0
        k1 = 0
        pos = 0
        
        for name, row in df[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act > 1e-9:
                P += 1.0
            else:
                N += 1.0

        if P==0:
            return 0.0
        if N==0:
            return 1.0

        for name, row in df[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act > 1e-9:
                k1 += 1
            else:
                k0 += 1
            tp1 = float(k1)/P
            fp1 = float(k0)/N
            auc += (fp1-fp0)*(tp1+tp0)/2
            #print("kk", (fp1-fp0)*(tp1+tp0)/2, fp1, fp0, tp1, tp0,k1, k0)
            tp0 = tp1
            fp0 = fp1
                
        return auc

class auc2(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        auc = 0.0
        k0 = 0
        k1 = 0
        tp0 = 0.0
        fp0 = 0.0
        tp1 = 1.0
        fp1 = 1.0
        P = 0
        N = 0
        ap = 0.0
        k0 = 0
        k1 = 0
        pos = 0

        for name, row in df[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act>1e-9:
                P += 1.0
            else:
                N += 1.0

        if P==0:
            return 0.0
        if N==0:
            return 1.0

        for name, row in df[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act > 1e-9:
                k1 += 1
                tp1 = float(k1)/P
                fp1 = float(k0)/N
                auc += (fp1-fp0)*(tp1+tp0)/2
                #print("kk", (fp1-fp0)*(tp1+tp0)/2, fp1, fp0, tp1, tp0, k1, k0)
                tp0 = tp1
                fp0 = fp1
            else:
                k0 += 1
                
        auc += 1.0 - fp1
        return auc

############compute auc juchi#############
class auc_zigzag(metric_base):
    def __init__(self, k=999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        rec = []
        for name, row in df[:self.k].iterrows():
            rec.append((row[s_label], row[self.s_score]))
    
        sum_pospair = 0.0
        sum_npos = 0.0
        sum_nneg = 0.0
        buf_pos = 0.0
        buf_neg = 0.0
        wt = 1
        for j in range(len(rec)):
            ctr = rec[j][0]
        #keep bucketing predictions in same bucket
            if j != 0 and rec[j][1] != rec[j - 1][1]:
                sum_pospair += buf_neg * (sum_npos + buf_pos *0.5)
                sum_npos += buf_pos    
                sum_nneg += buf_neg
                buf_neg = 0.0
                buf_pos = 0.0
          
            buf_pos += ctr * wt
            buf_neg += (1.0 - ctr) * wt
        
        sum_pospair += buf_neg * (sum_npos + buf_pos *0.5)
        sum_npos += buf_pos
        sum_nneg += buf_neg
        if sum_npos * sum_nneg == 0:
            return 0.5
        sum_auc = sum_pospair / (sum_npos*sum_nneg)
        return sum_auc

class dcg(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score
    
    def __call__(self, df):
        sumdcg = 0 
        i = 0 
        for name, row in df[:self.k].iterrows():
            val = row[self.s_score]
            act = row[s_label]
            if act > 0:
                sumdcg += ((1<<int(act))-1)*math.log(2)/math.log(2+i)
            i = i + 1 
        return sumdcg

class ndcg(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score
        
    def __call__(self, df):
        a = dcg()
        s1 = a(df)
        df2 = df.sort('label',ascending=False)
        s2 = a(df2)
        if s2 <= 1e-9:
            return 0
        else:
            return s1/s2


class rmse(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        return math.sqrt(np.mean([(row[self.s_score] - row[s_label])**2 for name, row in df[:self.k].iterrows()]))

class diff(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        return sum([(row[self.s_score] - row[s_label]) for name, row in df[:self.k].iterrows()])

class price(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        price_list = [row[s_price] for name, row in df[:self.k].iterrows()]
        return sum(price_list)*1.0/len(price_list)

class gmv(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        return sum([row[s_price] if row[s_label] > 0 else 0 for name, row in df[:self.k].iterrows()])


class click_avg_pos(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        click_list = []
        index = 0
        for name, row in df[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
               click_list.append(index)
        return sum(click_list)*1.0/len(click_list) if len(click_list)!=0 else 0

class order_avg_pos(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        order_list = []
        index = 0
        for name, row in df[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
               order_list.append(index) 
        return sum(order_list)*1.0/len(order_list) if len(order_list)!=0 else 0

class click_pos_dis(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        click_list = []
        index = 0
        for name, row in df[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
               click_list.append(index)
        return click_list

class order_pos_dis(metric_base):
    def __init__(self, k=99999999, s_score='score'):
        self.k = k
        self.s_score = s_score

    def __call__(self, df):
        order_list = []
        index = 0
        for name, row in df[:self.k].iterrows():
            index += 1
            if row[s_label] > 0:
               order_list.append(index)
        return order_list


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame([[3,1],[2,0],[1,1]], columns=['score','label'])
    
    
    m = ap(2,'score')
    print(m(df))


    a = auc()
    print(a(pd.DataFrame({'label':[1,1,0,1], 'score':[.26,.32,.52,.86]})), 1.0/3)
    print(a(pd.DataFrame({'label':[0,0,1,1,1], 'score':[.1,.1,.7,.8,.9]})), 0)
    print(a(pd.DataFrame({'label':[1,1,1,0,0], 'score':[.9,.8,.7,.1,.1]})), 1)
    print(a(pd.DataFrame({'label':[0,1,0,1], 'score':[.4,.3,.2,.1]})), 1.0/4)
    
    print("second")
    a = auc2()
    print(a(pd.DataFrame({'label':[1,1,0,1], 'score':[.26,.32,.52,.86]})), 1.0/3)
    print(a(pd.DataFrame({'label':[0,0,1,1,1], 'score':[.1,.1,.7,.8,.9]})), 0)
    a = rmse()
    print(a(pd.DataFrame({'label':[1,1,1,0,0], 'score':[.9,.8,.7,.1,.1]})), 1)
    
