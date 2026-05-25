"""
Назар бақылау модулі
DeepFace қайтаратын көз координаталары арқылы бет бағытын анықтайды
"""

import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, Any
import base64
from io import BytesIO
from PIL import Image


class AttentionDetector:
    """
    Бет бағытын анықтайтын класс.
    Көздер мен бет аумағының позициясын талдау арқылы:
    - focused (назарда) - камераға қарап тұр
    - distracted (алаңдаған) - сәл бұрылған
    - lost (назар жоқ) - басқа жаққа қарап тұр немесе бет көрінбейді
    """
    
    def __init__(self, detector_backend: str = 'opencv'):
        self.detector_backend = detector_backend
    
    def calculate_head_pose(self, face_region: Dict) -> Dict[str, float]:
        """
        Көз және бет координаталары арқылы бас бұрылысын есептеу
        Қайтарады:
            - yaw: солға (-) / оңға (+) бұрылу (-90..+90 градус)
            - pitch: бағалы шамалы есеп (көздердің биіктігі арқылы)
            - face_visible: бет толық көрініп тұр ма
        """
        if not face_region:
            return {
                'yaw': 0,
                'pitch': 0,
                'face_visible': False,
                'eyes_detected': False
            }
        
        x = face_region.get('x', 0)
        y = face_region.get('y', 0)
        w = face_region.get('w', 0)
        h = face_region.get('h', 0)
        
        left_eye = face_region.get('left_eye')
        right_eye = face_region.get('right_eye')
        
        # Бет ортасы
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        if not left_eye or not right_eye:
            return {
                'yaw': 0,
                'pitch': 0,
                'face_visible': True,
                'eyes_detected': False
            }
        
        # Көздер арасының ортасы
        eyes_center_x = (left_eye[0] + right_eye[0]) / 2
        eyes_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Yaw (солға/оңға бұрылу): көздер ортасы мен бет ортасының айырмасы
        # Егер көздер бет ортасынан солға қарай ауысса - оңға қараған
        offset_x = eyes_center_x - face_center_x
        yaw = (offset_x / (w / 2)) * 45  # шамамен ±45 градусқа дейін
        
        # Pitch (жоғары/төмен): көздердің беттегі вертикалды позициясы
        # Қалыпты жағдайда көздер беттің жоғарғы 40%-ында
        eye_relative_y = (eyes_center_y - y) / h
        # 0.35-0.45 — қалыпты, одан кіші — жоғары қарап тұр, үлкен — төмен
        pitch = (0.4 - eye_relative_y) * 100
        
        # Көздер бір-бірінен тым жақын болса — бет бүйірге бұрылған
        eye_distance = abs(right_eye[0] - left_eye[0])
        eye_distance_ratio = eye_distance / w if w > 0 else 0
        # Қалыпты қатынас ~0.35-0.45
        if eye_distance_ratio < 0.20:
            # Бет қатты бүйірге бұрылған
            yaw = yaw * 2 if abs(yaw) > 5 else (60 if yaw >= 0 else -60)
        
        return {
            'yaw': round(yaw, 2),
            'pitch': round(pitch, 2),
            'face_visible': True,
            'eyes_detected': True,
            'eye_distance_ratio': round(eye_distance_ratio, 3)
        }
    
    def determine_attention_status(self, head_pose: Dict) -> Dict[str, Any]:
        """
        Бас позициясы бойынша назар жағдайын анықтау
        Қайтарады:
            - status: 'focused' | 'distracted' | 'lost'
            - focus_score: 0-100 (фокус ұпайы)
            - direction: бағыты ('center', 'left', 'right', 'up', 'down')
        """
        if not head_pose['face_visible']:
            return {
                'status': 'lost',
                'status_kz': 'Бет көрінбейді',
                'focus_score': 0,
                'direction': 'none',
                'direction_kz': 'Жоқ',
                'message': 'Бет көрінбейді'
            }
        
        if not head_pose['eyes_detected']:
            return {
                'status': 'distracted',
                'status_kz': 'Көздер табылмады',
                'focus_score': 30,
                'direction': 'unknown',
                'direction_kz': 'Белгісіз',
                'message': 'Көздер айқын емес'
            }
        
        yaw = head_pose['yaw']
        pitch = head_pose['pitch']
        
        # Бағытты анықтау
        direction = 'center'
        direction_kz = 'Орталық'
        
        if abs(yaw) > abs(pitch):
            if yaw > 8:
                direction = 'right'
                direction_kz = 'Оңға'
            elif yaw < -8:
                direction = 'left'
                direction_kz = 'Солға'
        else:
            if pitch > 15:
                direction = 'up'
                direction_kz = 'Жоғары'
            elif pitch < -15:
                direction = 'down'
                direction_kz = 'Төмен'
        
        # Жалпы алаңдау өлшемі
        deviation = max(abs(yaw), abs(pitch) * 0.7)
        
        # Фокус ұпайы (100 - максимум, 0 - минимум)
        focus_score = max(0, min(100, 100 - deviation * 2))
        
        # Статус анықтау
        if deviation < 8:
            status = 'focused'
            status_kz = 'Назарда'
            message = 'Назарыңыз камерада'
        elif deviation < 20:
            status = 'distracted'
            status_kz = 'Алаңдау'
            message = 'Сәл алаңдау байқалады'
        else:
            status = 'lost'
            status_kz = 'Назар жоғалды'
            message = 'Назарыңыз басқа жаққа бөлінді'
        
        return {
            'status': status,
            'status_kz': status_kz,
            'focus_score': round(focus_score, 1),
            'direction': direction,
            'direction_kz': direction_kz,
            'message': message,
            'yaw': yaw,
            'pitch': pitch
        }
    
    def detect_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Суреттен назар жағдайын анықтау"""
        try:
            results = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False
            )
            
            if not results or len(results) == 0:
                return {
                    'success': True,
                    'face_detected': False,
                    'attention': {
                        'status': 'lost',
                        'status_kz': 'Бет табылмады',
                        'focus_score': 0,
                        'direction': 'none',
                        'direction_kz': 'Жоқ',
                        'message': 'Бет табылмады'
                    },
                    'region': {}
                }
            
            # Алғашқы бет
            face_data = results[0]
            region = face_data.get('facial_area', {})
            
            # Сенімділік төмен болса - бет жоқ деп есептеу
            confidence = face_data.get('confidence', 0)
            if confidence < 0.5:
                return {
                    'success': True,
                    'face_detected': False,
                    'attention': {
                        'status': 'lost',
                        'status_kz': 'Бет көрінбейді',
                        'focus_score': 0,
                        'direction': 'none',
                        'direction_kz': 'Жоқ',
                        'message': 'Бет анық емес'
                    },
                    'region': region
                }
            
            head_pose = self.calculate_head_pose(region)
            attention = self.determine_attention_status(head_pose)
            
            return {
                'success': True,
                'face_detected': True,
                'attention': attention,
                'head_pose': head_pose,
                'region': region,
                'confidence': round(confidence, 3)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'face_detected': False,
                'attention': {
                    'status': 'lost',
                    'status_kz': 'Қате',
                    'focus_score': 0,
                    'direction': 'none',
                    'direction_kz': 'Жоқ',
                    'message': str(e)
                }
            }
    
    def detect_from_base64(self, base64_string: str) -> Dict[str, Any]:
        """Base64 суреттен назар анықтау"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return self.detect_from_image(img_array)
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'face_detected': False,
                'attention': {
                    'status': 'lost',
                    'status_kz': 'Қате',
                    'focus_score': 0,
                    'direction': 'none',
                    'direction_kz': 'Жоқ',
                    'message': 'Сурет өңдеу қатесі'
                }
            }


if __name__ == "__main__":
    detector = AttentionDetector()
    print("Назар бақылау модулі дайын!")
    print("Статустар: focused, distracted, lost")
