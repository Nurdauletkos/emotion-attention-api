"""
Эмоция анықтау модулі
DeepFace кітапханасын қолданады - 7 негізгі эмоция + интенсивтілік
"""

import cv2
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image


class EmotionDetector:
    """Бет арқылы эмоция анықтайтын класс"""
    
    EMOTION_TRANSLATIONS = {
        'angry': 'Ашулы',
        'disgust': 'Жиіркеніш',
        'fear': 'Қорқыныш',
        'happy': 'Бақытты',
        'sad': 'Мұңды',
        'surprise': 'Таңқалу',
        'neutral': 'Бейтарап'
    }
    
    EMOTION_EMOJI = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😊',
        'sad': '😢',
        'surprise': '😮',
        'neutral': '😐'
    }
    
    def __init__(self, detector_backend: str = 'opencv'):
        """
        detector_backend опциялары:
        - 'opencv'     - ең жылдамы (real-time үшін)
        - 'mtcnn'      - орташа
        - 'retinaface' - ең дәлі (бірақ баяу)
        - 'mediapipe'  - жылдам әрі жақсы
        """
        self.detector_backend = detector_backend
    
    def get_intensity_label(self, score: float) -> str:
        """Интенсивтілік деңгейі (0-100)"""
        if score >= 75:
            return "Өте жоғары"
        elif score >= 50:
            return "Жоғары"
        elif score >= 25:
            return "Орташа"
        else:
            return "Төмен"
    
    def detect_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Суреттен барлық беттердің эмоциясын анықтау"""
        try:
            results = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results]
            
            output = []
            for face in results:
                emotions = face['emotion']
                dominant = face['dominant_emotion']
                intensity = float(emotions[dominant])
                
                sorted_emotions = sorted(
                    emotions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                all_emotions_kz = [
                    {
                        'emotion': emo,
                        'emotion_kz': self.EMOTION_TRANSLATIONS.get(emo, emo),
                        'emoji': self.EMOTION_EMOJI.get(emo, '😶'),
                        'score': round(float(score), 2)
                    }
                    for emo, score in sorted_emotions
                ]
                
                output.append({
                    'dominant_emotion': dominant,
                    'dominant_emotion_kz': self.EMOTION_TRANSLATIONS.get(dominant, dominant),
                    'emoji': self.EMOTION_EMOJI.get(dominant, '😶'),
                    'all_emotions': all_emotions_kz,
                    'intensity': round(intensity, 2),
                    'intensity_label': self.get_intensity_label(intensity),
                    'region': face.get('region', {})
                })
            
            return output
        
        except Exception as e:
            print(f"Қате: {e}")
            return []
    
    def detect_from_base64(self, base64_string: str) -> Dict[str, Any]:
        """Base64 кодталған суреттен эмоция анықтау"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = self.detect_from_image(img_array)
            
            return {
                'success': True,
                'faces_count': len(results),
                'faces': results
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'faces_count': 0,
                'faces': []
            }
    
    def detect_from_video(self, video_path: str, sample_rate: int = 30) -> Dict:
        """Видеодан эмоцияларды талдау"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Видео ашылмады'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        emotions_timeline = []
        emotion_counts = {emo: 0 for emo in self.EMOTION_TRANSLATIONS.keys()}
        intensity_sums = {emo: 0.0 for emo in self.EMOTION_TRANSLATIONS.keys()}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                results = self.detect_from_image(frame)
                if results:
                    timestamp = frame_idx / fps
                    for face in results:
                        emo = face['dominant_emotion']
                        emotions_timeline.append({
                            'timestamp': round(timestamp, 2),
                            'emotion': emo,
                            'emotion_kz': face['dominant_emotion_kz'],
                            'emoji': face['emoji'],
                            'intensity': face['intensity']
                        })
                        emotion_counts[emo] += 1
                        intensity_sums[emo] += face['intensity']
            
            frame_idx += 1
        
        cap.release()
        
        total = sum(emotion_counts.values())
        emotion_summary = []
        if total > 0:
            for emo, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    avg_intensity = intensity_sums[emo] / count
                    emotion_summary.append({
                        'emotion': emo,
                        'emotion_kz': self.EMOTION_TRANSLATIONS.get(emo, emo),
                        'emoji': self.EMOTION_EMOJI.get(emo, '😶'),
                        'count': count,
                        'percentage': round((count / total) * 100, 2),
                        'avg_intensity': round(avg_intensity, 2)
                    })
        
        return {
            'success': True,
            'duration_seconds': round(total_frames / fps, 2),
            'total_frames': total_frames,
            'analyzed_frames': len(emotions_timeline),
            'emotion_summary': emotion_summary,
            'timeline': emotions_timeline
        }


if __name__ == "__main__":
    detector = EmotionDetector()
    print("Эмоция анықтау модулі дайын!")
    print(f"Эмоциялар: {list(detector.EMOTION_TRANSLATIONS.values())}")
