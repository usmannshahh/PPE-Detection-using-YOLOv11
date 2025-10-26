"""
PPE Detection Module
YOLOv11 based Construction Site Safety Detection
Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, 
         Safety Cone, Safety Vest, machinery, vehicle
"""

import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import os


class PPEDetector:
    def __init__(self, model_path='best.pt'):
        """Initialize YOLOv11 model"""
        print(f"🔍 Loading YOLOv11 model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"📝 Classes: {self.model.names}")
        
        # Class names
        self.class_names = self.model.names
        
        # Violation classes (will trigger alarm)
        self.violation_classes = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
        
        print(f"⚠️ Monitoring violations: {self.violation_classes}")
    
    def is_violation(self, class_name):
        """Check if class is a safety violation"""
        return class_name in self.violation_classes
    
    def detect_video(self, video_path, output_path, alarm_callback=None):
        """Detect PPE violations in video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Output video with multiple codec fallback
        out = None
        codecs = [
            ('avc1', '.mp4'),  # H.264 - Best for browser
            ('H264', '.mp4'),  # Alternative H.264
            ('mp4v', '.mp4'),  # MPEG-4
            ('XVID', '.avi'),  # Fallback to AVI
        ]
        
        for codec, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_output = output_path.replace('.mp4', ext).replace('.avi', ext)
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                
                if out.isOpened():
                    output_path = temp_output
                    print(f"✅ Using codec: {codec} -> {Path(output_path).name}")
                    break
                else:
                    out.release()
            except Exception as e:
                print(f"⚠️ Codec {codec} failed: {e}")
                continue
        
        if out is None or not out.isOpened():
            raise Exception("❌ Could not initialize video writer with any codec!")
        
        frame_count = 0
        violation_frames = 0
        total_violations = 0
        violation_detected = False
        
        # Stats
        violation_stats = {
            'NO-Hardhat': 0,
            'NO-Mask': 0,
            'NO-Safety Vest': 0
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            results = self.model.predict(
                frame,
                conf=0.4,
                iou=0.45,
                verbose=False
            )
            
            # Annotated frame
            annotated_frame = results[0].plot()
            
            # Check violations
            frame_violations = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.class_names[cls_id]
                    confidence = float(box.conf[0])
                    
                    if self.is_violation(class_name):
                        violation_detected = True
                        total_violations += 1
                        violation_stats[class_name] += 1
                        frame_violations.append(class_name)
                        
                        if alarm_callback:
                            alarm_callback()
                        
                        print(f"⚠️ Frame {frame_count}: {class_name} (conf: {confidence:.2f})")
            
            # Add warning overlay
            if frame_violations:
                violation_frames += 1
                
                # Red warning box
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, 10), (width-10, 120), (0, 0, 200), -1)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
                
                # Warning text
                cv2.putText(annotated_frame, '🚨 SAFETY VIOLATION!',
                           (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                
                violation_text = ', '.join(set(frame_violations))
                cv2.putText(annotated_frame, f'Violations: {violation_text}',
                           (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Frame counter
            cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}',
                       (width - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(annotated_frame)
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # Verify output file
        if not Path(output_path).exists():
            raise Exception(f"❌ Output video not created: {output_path}")
        
        file_size = Path(output_path).stat().st_size / (1024*1024)
        print(f"📁 Output file size: {file_size:.2f} MB")
        
        print(f"\n{'='*60}")
        print(f"✅ Detection Complete!")
        print(f"📊 Frames: {frame_count} | Violations: {violation_frames}")
        print(f"🔢 Total detections: {total_violations}")
        for vtype, count in violation_stats.items():
            if count > 0:
                print(f"   • {vtype}: {count}")
        print(f"{'='*60}\n")
        
        return {
            'frames_processed': frame_count,
            'violations_detected': violation_detected,
            'violation_frames': violation_frames,
            'violation_count': total_violations,
            'violation_stats': violation_stats,
            'output_path': output_path
        }
    
    def detect_frame(self, frame):
        """Detect PPE violations in single frame (for live camera)"""
        results = self.model.predict(frame, conf=0.4, iou=0.45, verbose=False)
        annotated_frame = results[0].plot()
        
        violation = False
        violations_list = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                confidence = float(box.conf[0])
                
                if self.is_violation(class_name):
                    violation = True
                    violations_list.append({'class': class_name, 'confidence': confidence})
        
        # Add warning overlay
        if violation:
            h, w = annotated_frame.shape[:2]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 200), -1)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
            
            cv2.putText(annotated_frame, '🚨 SAFETY VIOLATION!',
                       (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
            
            violation_names = [v['class'] for v in violations_list]
            violation_text = ', '.join(set(violation_names))
            cv2.putText(annotated_frame, violation_text[:50],
                       (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, violation
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'YOLOv11',
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'violation_classes': self.violation_classes,
            'task': 'detect'
        }


# Test code
if __name__ == "__main__":
    print("Testing PPE Detector...")
    detector = PPEDetector('best.pt')
    print("✅ Test successful!")