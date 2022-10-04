import csv
import argparse
from functools import total_ordering

def read_csv(path):
    label = []
    with open(f'{path}', newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        for i, row in enumerate(rows):
            if i==0: 
                continue
            filename, l = row
            label.append(l)

    return label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 eval",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pred_path", help="Predicted csv location")
    parser.add_argument("gth_path", help="Ground truth csv location")
    args = parser.parse_args()

    pred_path = args.pred_path
    gth_path = args.gth_path

    pred = read_csv(pred_path)
    gth = read_csv(gth_path)
    print(len(pred))
    print(len(gth))
    hit = 0
    total_cnt = 0
    for i in range(len(gth)):
        if pred[i] == gth[i]:
            hit=hit+1
        total_cnt=total_cnt+1
    print(total_cnt)
    print("Acc:", hit / total_cnt)





