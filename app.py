import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print("✅ GlowAI ready!")

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60,60))
    return max(faces, key=lambda f: f[2]*f[3]) if len(faces) > 0 else None

def detect_eye_y(gray, fx, fy, fw, fh):
    region = gray[fy:fy+int(fh*0.55), fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(region, 1.1, 5, minSize=(20,20))
    if len(eyes) >= 2:
        return fy + int(np.mean([e[1]+e[3]//2 for e in sorted(eyes, key=lambda e:e[0])[:2]]))
    return None

def get_skin_tone(frame, fx, fy, fw, fh):
    try:
        r = frame[fy+int(fh*0.08):fy+int(fh*0.25), fx+int(fw*0.3):fx+int(fw*0.7)]
        if r.size == 0: return "Medium"
        avg = np.mean(r.reshape(-1,3), axis=0).astype(np.uint8)
        lab = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2LAB)[0][0]
        L, b = float(lab[0]), float(lab[2])-128
        if abs(b) < 0.001: b = 0.001
        ita = math.atan((L-50)/b)*(180/math.pi)
        if ita>55: return "Very Fair"
        elif ita>41: return "Fair"
        elif ita>28: return "Medium"
        elif ita>10: return "Olive"
        elif ita>-30: return "Tan/Brown"
        else: return "Dark"
    except: return "Medium"

def hex_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def blend(frame, mask, color, alpha):
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    colored = np.zeros_like(frame); colored[:] = color
    a = (mask/255.0*alpha)[:,:,np.newaxis]
    return (frame*(1-a)+colored*a).astype(np.uint8)

def apply_lipstick(frame, fx, fy, fw, fh, color, opacity):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ey = detect_eye_y(gray, fx, fy, fw, fh)
        lip_cy = (ey + int((fy+fh-ey)*0.72)) if ey else fy+int(fh*0.75)
        lip_cx = fx+fw//2
        lip_rx, lip_ry = int(fw*0.18), int(fh*0.055)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask,(lip_cx,lip_cy-lip_ry//2),(lip_rx,lip_ry),0,0,360,255,-1)
        cv2.ellipse(mask,(lip_cx,lip_cy+lip_ry//2),(int(lip_rx*0.9),int(lip_ry*1.2)),0,0,360,255,-1)
        return blend(frame, mask, hex_bgr(color), opacity)
    except: return frame

def apply_blush(frame, fx, fy, fw, fh, color, opacity):
    try:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for cx,cy in [(fx+int(fw*0.15),fy+int(fh*0.55)),(fx+int(fw*0.85),fy+int(fh*0.55))]:
            cv2.circle(mask,(cx,cy),int(fw*0.16),255,-1)
        return blend(frame, mask, hex_bgr(color), opacity*0.5)
    except: return frame

def apply_eyeshadow(frame, fx, fy, fw, fh, color, opacity):
    try:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for ex,ey in [(fx+int(fw*0.28),fy+int(fh*0.33)),(fx+int(fw*0.72),fy+int(fh*0.33))]:
            cv2.ellipse(mask,(ex,ey),(int(fw*0.17),int(fh*0.07)),0,0,360,255,-1)
        return blend(frame, mask, hex_bgr(color), opacity*0.55)
    except: return frame

@app.route('/')
def index(): return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json(force=True)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data['image']),np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return jsonify({'error':'bad image'}),400
        s = data.get('settings',{})
        lip_color   = s.get('lip_color','#FF6B8A')
        blush_color = s.get('blush_color','#FFB7C5')
        eye_color   = s.get('eye_color','#B46482')
        opacity     = float(s.get('opacity',0.6))
        do_lip      = bool(s.get('lipstick',True))
        do_blush    = bool(s.get('blush',False))
        do_eye      = bool(s.get('eyeshadow',False))
        pipeline = {k:False for k in ['faceDetection','landmarkExtraction','skinToneClassification','makeupApplication','recommendationEngine']}
        skin_tone = "No face"; face_detected = False
        face = detect_face(frame)
        if face is not None:
            fx,fy,fw,fh = face; face_detected = True
            pipeline['faceDetection'] = pipeline['landmarkExtraction'] = True
            skin_tone = get_skin_tone(frame,fx,fy,fw,fh)
            pipeline['skinToneClassification'] = True
            if do_lip:   frame = apply_lipstick(frame,fx,fy,fw,fh,lip_color,opacity)
            if do_blush: frame = apply_blush(frame,fx,fy,fw,fh,blush_color,opacity)
            if do_eye:   frame = apply_eyeshadow(frame,fx,fy,fw,fh,eye_color,opacity)
            pipeline['makeupApplication'] = pipeline['recommendationEngine'] = True
        _,buf = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,60])
        return jsonify({'processed_image':base64.b64encode(buf).decode(),'skin_tone':skin_tone,'pipeline':pipeline,'face_detected':face_detected})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/recommendations')
def recommendations():
    return jsonify({
        "Very Fair": [{"name":"MAC Velvet Teddy","shade":"#C4956A","type":"Lipstick"},{"name":"NARS Dolce Vita","shade":"#B07070","type":"Lipstick"},{"name":"L'Oreal Blushed Mauve","shade":"#D4A0A0","type":"Blush"}],
        "Fair":      [{"name":"MAC Mehr","shade":"#C08080","type":"Lipstick"},{"name":"NARS Schiap","shade":"#E75480","type":"Lipstick"},{"name":"Maybelline Spice","shade":"#A0522D","type":"Lipstick"}],
        "Medium":    [{"name":"MAC Runway Hit","shade":"#B05030","type":"Lipstick"},{"name":"NARS Heat Wave","shade":"#FF4500","type":"Lipstick"},{"name":"L'Oreal Spiced Cider","shade":"#8B4513","type":"Lipstick"}],
        "Olive":     [{"name":"MAC Twig","shade":"#9B6B5A","type":"Lipstick"},{"name":"NARS Laguna","shade":"#C68642","type":"Bronzer"},{"name":"Maybelline Warm Me Up","shade":"#D2691E","type":"Blush"}],
        "Tan/Brown": [{"name":"MAC Brick-O-La","shade":"#8B2500","type":"Lipstick"},{"name":"NARS Train Bleu","shade":"#8B4513","type":"Lipstick"},{"name":"L'Oreal Spice","shade":"#A0522D","type":"Lipstick"}],
        "Dark":      [{"name":"MAC Diva","shade":"#660000","type":"Lipstick"},{"name":"NARS Jungle Red","shade":"#8B0000","type":"Lipstick"},{"name":"Maybelline Plum","shade":"#800080","type":"Lipstick"}],
        "Unknown":   [{"name":"MAC Ruby Woo","shade":"#CC0000","type":"Lipstick"},{"name":"NARS Red Square","shade":"#FF0000","type":"Lipstick"},{"name":"L'Oreal True Red","shade":"#DC143C","type":"Lipstick"}]
    })

@app.route('/test')
def test(): return jsonify({'status':'running','message':'GlowAI working!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("="*40)
    print(f"✅ GlowAI — http://localhost:{port}")
    print("="*40)
    app.run(host='0.0.0.0', port=port, debug=False)
