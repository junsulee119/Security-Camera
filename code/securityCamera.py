import numpy as np
import cv2 as cv
import time as tm 
from datetime import datetime

video_name = "Laptop Webcam"
video_source_path = 0
video_save_format = "avi"
video_save_format_fourcc = "XVID"

isRecording = False
isSentrymode = False
recorded_video = None

# 움직임 감지 관련 파라미터
img_back = None
motion_detected = False
blur_ksize = (9, 9)
blur_sigma = 3
diff_threshold = 150
bg_update_rate = 0.05
fg_update_rate = 0.001

# 센트리모드 초기화 관련 변수 추가
collecting_background = False
frame_count = 0
background_frames = []

def draw_camera_frame(img, w, h):
    # 좌상단 프레임
    cv.line(img, (int(w / 20), int(h / 20)), (int(w / 20), int(h*5 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w / 20), int(h / 20)), (int(w / 20), int(h*5 / 20)), (0, 0, 0))
    cv.line(img, (int(w / 20), int(h / 20)), (int(w*5 / 20), int(h / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w / 20), int(h / 20)), (int(w*5 / 20), int(h / 20)), (0, 0, 0))
    # 좌하단 프레임
    cv.line(img, (int(w / 20), int(h*19 / 20)), (int(w / 20), int(h*15 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w / 20), int(h*19 / 20)), (int(w / 20), int(h*15 / 20)), (0, 0, 0))
    cv.line(img, (int(w / 20), int(h*19 / 20)), (int(w*5 / 20), int(h*19 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w / 20), int(h*19 / 20)), (int(w*5 / 20), int(h*19 / 20)), (0, 0, 0))
    # 우상단 프레임
    cv.line(img, (int(w*19 / 20), int(h / 20)), (int(w*19 / 20), int(h*5 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w*19 / 20), int(h / 20)), (int(w*19 / 20), int(h*5 / 20)), (0, 0, 0))
    cv.line(img, (int(w*19 / 20), int(h / 20)), (int(w*15 / 20), int(h / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w*19 / 20), int(h / 20)), (int(w*15 / 20), int(h / 20)), (0, 0, 0))
    # 우하단 프레임
    cv.line(img, (int(w*19 / 20), int(h*19 / 20)), (int(w*19 / 20), int(h*15 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w*19 / 20), int(h*19 / 20)), (int(w*19 / 20), int(h*15 / 20)), (0, 0, 0))
    cv.line(img, (int(w*19 / 20), int(h*19 / 20)), (int(w*15 / 20), int(h*19 / 20)), (255, 255, 255), thickness=2)
    cv.line(img, (int(w*19 / 20), int(h*19 / 20)), (int(w*15 / 20), int(h*19 / 20)), (0, 0, 0))

def start_record(video, img, fps):
    global isRecording, recorded_video
    isRecording = True
    if recorded_video is None:
        recorded_video = cv.VideoWriter()
        recorded_video_name = tm.ctime().replace(":", "").replace(" ", "_") + "." + video_save_format
        h, w, *_ = img.shape
        is_color = (img.ndim > 2) and (img.shape[2] > 1)
        fourcc = cv.VideoWriter_fourcc(*video_save_format_fourcc)
        success = recorded_video.open(recorded_video_name, fourcc, fps, (w, h), is_color)
        if not success:
            print("Failed to initialize video writer")
            recorded_video = None
            isRecording = False

def end_record():
    global isRecording, recorded_video
    if recorded_video is not None:
        recorded_video.release()
        recorded_video = None
    isRecording = False

def start_sentrymode(video, img):
    global isSentrymode, collecting_background, frame_count, background_frames, img_back
    isSentrymode = True
    collecting_background = True
    frame_count = 0
    background_frames = []

    # 카운트다운 시작
    countdown_start = datetime.now()
    while (datetime.now() - countdown_start).total_seconds() < 5:
        valid, countdown_img = video.read()
        if not valid:
            break
        
        # 카운트다운 계산
        elapsed = (datetime.now() - countdown_start).total_seconds()
        countdown_num = 5 - int(elapsed)
        
        # 안내 메시지 및 카운트다운 표시
        h, w = countdown_img.shape[:2]
        text = f"Sentry mode starting in {countdown_num}"
        cv.putText(countdown_img, text, (int(w*5 / 20), int(h * 10 / 20)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(countdown_img, text, (int(w*5 / 20), int(h * 10 / 20)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
        cv.putText(countdown_img, "Try to remove all the movable objects!", (int(w*2 / 20), int(h * 13 / 20)), \
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(countdown_img, "Try to remove all the movable objects!", (int(w*2 / 20), int(h * 13 / 20)), \
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        cv.imshow(f"Live Video from {video_name}", countdown_img)
        if cv.waitKey(50) == 27:
            end_sentrymode()
            return

    # 백그라운드 프레임 수집 시작
    collecting_start = datetime.now()
    while len(background_frames) < 1000:
        valid, bg_img = video.read()
        if not valid:
            break
        
        # 프레임 처리
        img_blur = cv.GaussianBlur(bg_img, blur_ksize, blur_sigma)
        background_frames.append(img_blur)
        
        # 진행 상태 표시
        frame_count += 1
        h, w = bg_img.shape[:2]
        status_img = bg_img.copy()
        cv.putText(status_img, f"Collecting frames: {frame_count}/1000", (int(w*4 / 20), int(h*9 / 20)), \
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(status_img, f"Collecting frames: {frame_count}/1000", (int(w*4 / 20), int(h*9 / 20)), \
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
        cv.putText(status_img, "Initializing motion detection...", (int(w*4 / 20), int(h*11 / 20)), \
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv.putText(status_img, "Initializing motion detection...", (int(w*4 / 20), int(h*11 / 20)), \
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
        
        cv.imshow(f"Live Video from {video_name}", status_img)
        if cv.waitKey(1) == 27:
            end_sentrymode()
            return

    # 백그라운드 평균 계산
    if background_frames:
        img_back = np.mean(background_frames, axis=0).astype(np.float64)
    collecting_background = False

def end_sentrymode():
    global isSentrymode, motion_detected
    isSentrymode = False
    motion_detected = False
    # 센트리 모드 종료 시 녹화 중이었다면 녹화 종료
    if isRecording:
        end_record()

def detect_motion(img):
    global img_back, motion_detected
    
    # 움직임 감지 로직을 위한 필터 함수
    box = lambda ksize: np.ones((ksize, ksize), dtype=np.uint8)
    
    # 현재 프레임을 블러 처리
    img_blur = cv.GaussianBlur(img, blur_ksize, blur_sigma)
    
    # 배경과의 차이 계산
    img_diff = img_blur - img_back
    
    # 차이의 놂(거리) 계산
    img_norm = np.linalg.norm(img_diff, axis=2)
    
    # 이진화
    img_bin = np.zeros_like(img_norm, dtype=np.uint8)
    img_bin[img_norm > diff_threshold] = 255
    
    # 노이즈 제거 및 마스크 생성
    img_mask = img_bin.copy()
    img_mask = cv.erode(img_mask, box(3))
    img_mask = cv.dilate(img_mask, box(5))
    img_mask = cv.dilate(img_mask, box(3))
    
    # 전경 픽셀 판별
    fg = img_mask == 255
    
    # 배경 픽셀 판별
    bg = ~fg
    
    # 배경 모델 업데이트
    img_back[bg] = (bg_update_rate * img_blur[bg] + (1 - bg_update_rate) * img_back[bg])
    img_back[fg] = (fg_update_rate * img_blur[fg] + (1 - fg_update_rate) * img_back[fg])
    
    # 움직임이 감지되었는지 확인 (전경 픽셀이 임계값 이상인 경우)
    motion_detected = np.sum(fg) > 100  # 임계값은 필요에 따라 조정 가능
    
    return img_mask, motion_detected

if __name__ == "__main__":
    video = cv.VideoCapture(video_source_path)
    
    if video.isOpened():
        # FPS 측정
        num_frames = 0
        start_time = tm.time()
        while num_frames < 30:
            valid, img = video.read()
            if valid:
                num_frames += 1
        measured_fps = num_frames / (tm.time() - start_time)
        actual_fps = measured_fps if measured_fps > 0 else 30
        wait_msec = int(1 / actual_fps * 1000) if actual_fps > 0 else 30
        
        while True:
            valid, img = video.read()
            if not valid:
                break
            
            if collecting_background:
                continue
                
            h, w, *_ = img.shape
            
            img_display = img.copy()
            draw_camera_frame(img_display, w, h)
            
            if isSentrymode:
                if img_back is not None:
                    img_mask, motion = detect_motion(img)
                    
                    if motion and not isRecording:
                        start_record(video, img, actual_fps)
                    elif not motion and isRecording:
                        end_record()
                        
                    motion_text = "Motion Detected!" if motion else "No Motion"
                    cv.putText(img_display, motion_text, (int(w*10 / 20), int(h*19 / 20)), cv.FONT_HERSHEY_DUPLEX, 0.8, 
                              (255, 255, 255), thickness=3)
                    cv.putText(img_display, motion_text, (int(w*10 / 20), int(h*19 / 20)), cv.FONT_HERSHEY_DUPLEX, 0.8, 
                              (0, 0, 255) if motion else (0, 255, 0), thickness=2)
                
                cv.putText(img_display, "Sentry Mode Active", (int(w*10 / 20), int(h*17.5 / 20)), cv.FONT_HERSHEY_DUPLEX, 0.8, 
                          (255, 255, 255), thickness=3)
                cv.putText(img_display, "Sentry Mode Active", (int(w*10 / 20), int(h*17.5 / 20)), cv.FONT_HERSHEY_DUPLEX, 0.8, 
                          (88, 88, 88), thickness=2)
            
            if isRecording:
                recorded_video.write(img)
                cv.putText(img_display, "REC", (int(w*2.5 / 20), int(h*2 / 20)), cv.FONT_HERSHEY_DUPLEX, 0.7, 
                          (255, 255, 255), thickness=2)
                cv.putText(img_display, "REC", (int(w*2.5 / 20), int(h*2/ 20)), cv.FONT_HERSHEY_DUPLEX, 0.7, 
                          (0, 0, 255))
                cv.circle(img_display, (int(w*2 / 20), int(h*1.7 / 20)), 5, (0, 0, 255), -1)
            
            cv.imshow(f"Live Video from {video_name}", img_display)
            
            if isSentrymode and img_back is not None:
                img_back_display = img_back.astype(np.uint8)
                cv.imshow("Background Model", img_back_display)
            
            key = cv.waitKey(wait_msec)
            if key == 27:
                break
            elif key == ord(' '):
                if not isSentrymode:
                    if isRecording:
                        end_record()
                    else:
                        start_record(video, img, actual_fps)
            elif key == ord('s') or key == ord('S'):
                if isSentrymode:
                    end_sentrymode()
                else:
                    start_sentrymode(video, img)

        end_record()
        cv.destroyAllWindows()
        video.release()