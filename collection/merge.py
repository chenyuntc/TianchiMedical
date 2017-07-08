import csv


def mer(f1,f2,of):
    with open(f1) as f:
        lines1 = f.readlines()[1:]
        lines1 = [_.split(',')[:-1] for _ in lines1]
        lines1 = sorted(lines1,key=lambda x:float(x[4]),reverse=True)
        
        lines1 = lines1[:2000]
    with open(f2) as f:
        lines2 = f.readlines()[1:]
        lines2 = [_.split(',')[:-1] for _ in lines2]
        lines2 = sorted(lines2,key=lambda x:float(x[4]),reverse=True)
        lines2 = lines2[:2000]


    lines1 = lines1[:400]
    
    
    
    lines1_ids = { (_[0],_[1]) for _ in lines1 }
    lines2_filter = [ _ for _ in lines2 if (_[0],_[1]) not in lines1_ids]
    # import ipdb;ipdb.set_trace() 
    lines2 = lines2_filter[:1500]
    for line in lines2:
        line[4]=0.5*float(line[4])
    for line in lines1:
        line[4] = 0.5 +0.5*float(line[4])
    all_lines = lines1 + lines2 
    with  open(of,'w')  as f:
        writer = csv.writer(f)
        writer.writerow(
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability','isnodule','diameter_mm'] )
        writer.writerows(all_lines)