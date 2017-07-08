#coding:utf8
def calculage_froc(results,total_right_num,file_num):
    '''
    @results:everyline in csv
    @total_right_num: 总共标记的节点数
    
    line: serid,x,y,z,probability,isnode 

    !TODO:解决多个结果对应识别为同一个标注的问题
    '''
    wrong = [0 for _ in results]
    right = [0 for _ in results]
    wrongnum ,rightnum = 0,0
    for (ii,result) in enumerate(results):
        if result[-1]==1:
            rightnum+=1
        else :
            wrongnum+=1
        wrong[ii]=wrongnum
        right[ii] = rightnum
    
    
    cal_point = [0.125,0.25,0.5,1,2,4,8]
    scores = []
    for point in cal_point:
        wrong_num_limit = int(file_num*point) 
        index = wrong.index(wrong_num_limit)
        scores.append( float(right[index])/total_right_num)
        print right[index]
    print scores
    
    return sum(scores)/len(scores)

def main(path,total_num = 1244,file_num=721):
    '''
    usage: python cal_froc.py main a.csv 1000 100
    '''
    with open(path) as f:
        lines = f.readlines()[1:]
    lines = [line.strip().split(',') for line in lines]
    # results = [[s,x,y,z,float(p),int(float(i))] for [s,x,y,z,p,i,_]  in lines ]
    results = [[line[0],float(line[4]),int(float(line[5]))] for line  in lines ]
    results = sorted(results,key=lambda x:x[-2],reverse=True)
    froc = calculage_froc(results,total_num,file_num)
    print(froc)
    return froc

if __name__=='__main__':
    import fire
    fire.Fire()
