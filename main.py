import os
import torch
import pandas
import subprocess


symbol_to_letter = {'one': '11', 'two': '12', 'three': '13', 'four': '14', 'five': '15', 'six': '16', 'seven': '17', 'eight': '18', 'nine': '19', 'A': '20', 'B': '21', 'C': '22', 'D': '23', 'E': '24', 'F': '25', 'G': '26', 'H': '27', 'S': '28', 'T': '29', 'U': '30', 'V': '31', 'W': '32', 'X': '33', 'Y': '34', 'Z': '35', 'left': '39', 'up': '36', 'right': '38', 'down': '37', 'circle': '40', 'bulleye': '0'}
output_path = 'yolov5/runs/detect/exp3'
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt', force_reload=True)

def run():
    # command = f'python3 yolov5/detect.py --weights best.pt --img 416 --conf 0.1 --source ./test/images --nosave'
    command = f'python3 yolov5/detect.py --weights best.pt --img 416 --conf 0.1 --source ./test/images2'
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    print(output)

    # exp_path = os.path.join(os.getcwd(), output_path)
    # for file_path in os.listdir(exp_path):
    #     abs_path = os.path.join(exp_path,file_path)
    #     get_inference(abs_path)

def get_inference(file_path):
    results = model(file_path)
    label = ''
    try:
        data = results.pandas().xyxy[0].to_dict(orient = 'records')[0]
        print(file_path,data['name'])
    except:
        print('an error occured')
    return label

if __name__ == '__main__':
    run()