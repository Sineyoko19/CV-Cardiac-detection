import cv2
import torch
import pydicom
import numpy as np
from typing import Dict

class CardiacRiskEvaluation:
    def __init__(self):
        self.risk_weights = {
            "ctr": 0.40,
            "position": 0.35,
            "vertical": 0.25
        }

    def extract_chest_region(self, img_tensor: torch.Tensor) -> Dict:
        """
        Automatically detect thoracic cavity boundaries
        """
        
        height, width = img_tensor.shape[2:]
        
        # Standard positioning for PA/AP chest X-rays
        chest_bbox = {
            'x_min': int(width * 0.05),
            'x_max': int(width * 0.95),
            'y_min': int(height * 0.10),
            'y_max': int(height * 0.85),
            'center_x': width // 2,
            'center_y': int(height * 0.50)
        }
        
        return chest_bbox
    
    def calculate_ctr(self, heart_bbox: list, chest_bbox: Dict) -> Dict:
        """
        Calculate Cardiothoracic Ratio
        """
        heart_width = heart_bbox[2]
        chest_width = chest_bbox['x_max'] - chest_bbox['x_min']
        
        ctr = heart_width / chest_width
        
        return {
            'ctr': ctr,
            'heart_width_px': heart_width,
            'chest_width_px': chest_width,
            'chest_bbox': chest_bbox
        }
    
    def assess_ctr_risk(self, ctr: float) -> Dict:
        """Risk assessment based on CTR"""
        if ctr < 0.42:
            return {"risk": "low", "finding": "Small heart", "score": 1}
        elif 0.42 <= ctr <= 0.50:
            return {"risk": "low", "finding": "Normal heart size", "score": 0}
        elif 0.50 < ctr <= 0.55:
            return {"risk": "moderate", "finding": "Borderline cardiomegaly", "score": 2}
        else:
            return {"risk": "high", "finding": "Cardiomegaly", "score": 3}
    
    def assess_cardiac_position(self, heart_bbox: list, chest_bbox: Dict) -> Dict:
        """
        Assess horizontal cardiac position
        """
        
        x = heart_bbox[0]
        y = heart_bbox[1]
        w = heart_bbox[2]
        h = heart_bbox[3]
        
        heart_center_x = x + w // 2

        chest_center_x = chest_bbox['center_x']
        chest_width = chest_bbox['x_max'] - chest_bbox['x_min']
        
        # Expected heart position: ~7% left of center
        expected_heart_x = chest_center_x - (chest_width * 0.07)
        deviation = heart_center_x - expected_heart_x
        deviation_percent = (deviation / chest_width) * 100
        
        # Also calculate deviation from midline
        midline_deviation = heart_center_x - chest_center_x
        midline_deviation_percent = (midline_deviation / chest_width) * 100
        
        # Assess position
        if heart_center_x > chest_center_x + (chest_width * 0.05):
            return {
                "risk": "high",
                "finding": "Dextrocardia or significant right shift",
                "score": 3,
                "deviation_percent": deviation_percent,
                "midline_deviation_percent": midline_deviation_percent,
                "laterality": "right"
            }
        elif heart_center_x > chest_center_x - (chest_width * 0.02):
            return {
                "risk": "moderate",
                "finding": "Heart more centered than typical",
                "score": 2,
                "deviation_percent": deviation_percent,
                "midline_deviation_percent": midline_deviation_percent,
                "laterality": "centered"
            }
        elif heart_center_x < chest_center_x - (chest_width * 0.20):
            return {
                "risk": "moderate",
                "finding": "Extreme left shift - possible rotation or mediastinal shift",
                "score": 2,
                "deviation_percent": deviation_percent,
                "midline_deviation_percent": midline_deviation_percent,
                "laterality": "extreme_left"
            }
        else:
            return {
                "risk": "low",
                "finding": "Normal left-sided cardiac position",
                "score": 0,
                "deviation_percent": deviation_percent,
                "midline_deviation_percent": midline_deviation_percent,
                "laterality": "normal_left"
            }
    def assess_vertical_position(self, heart_bbox: list, chest_bbox: Dict) -> Dict:
        """
        Assess vertical cardiac position
        """
        x = heart_bbox[0]
        y = heart_bbox[1]
        w = heart_bbox[2]
        h = heart_bbox[3]

        heart_center_y = y + h // 2

        chest_height = chest_bbox['y_max'] - chest_bbox['y_min']
        chest_top = chest_bbox['y_min']
        
        relative_position = (heart_center_y - chest_top) / chest_height
        
        if relative_position < 0.35:
            return {
                "risk": "moderate",
                "finding": "Heart elevated (possible hyperinflation/COPD)",
                "score": 2,
                "relative_position": relative_position
            }
        elif relative_position > 0.75:
            return {
                "risk": "moderate",
                "finding": "Heart descended (possible effusion/mass/diaphragm issue)",
                "score": 2,
                "relative_position": relative_position
            }
        else:
            return {
                "risk": "low",
                "finding": "Normal vertical position",
                "score": 0,
                "relative_position": relative_position
            }
        
    def comprehensive_assessment(self, 
                                 img_tensor: torch.Tensor, 
                                 heart_bbox: list) -> Dict:
        """
        Complete assessment from DICOM file
        
        Args:
            dicom_path: Path to .dcm file
            heart_bbox: Detected heart bounding box
            use_lung_detection: Try to detect lung boundaries (experimental)
        """
        # Load DICOM
        results = {}
        chest_bbox = self.extract_chest_region(img_tensor)
        
        # Calculate CTR
        ctr_data = self.calculate_ctr(heart_bbox, chest_bbox)
        results['ctr'] = self.assess_ctr_risk(ctr_data['ctr'])
        results['ctr']['value'] = ctr_data['ctr']
        results['ctr']['heart_width_px'] = ctr_data['heart_width_px']
        results['ctr']['chest_width_px'] = ctr_data['chest_width_px']
  
        # Assess position
        results['position'] = self.assess_cardiac_position(heart_bbox, chest_bbox)
        
        # Assess vertical position
        results['vertical'] = self.assess_vertical_position(heart_bbox, chest_bbox)
        
        # Calculate weighted score
        total_score = (
            results['ctr']['score'] * self.risk_weights['ctr'] +
            results['position']['score'] * self.risk_weights['position'] +
            results['vertical']['score'] * self.risk_weights['vertical']
        )
        
        # Overall classification
        if total_score < 0.5:
            overall_risk = "LOW"
            recommendation = "No significant abnormalities detected in cardiac position and size"
        elif total_score < 1.5:
            overall_risk = "MODERATE"
            recommendation = "Minor findings detected - clinical correlation recommended"
        else:
            overall_risk = "HIGH"
            recommendation = "Significant findings - further evaluation strongly recommended"
        
        return {
            "overall_risk": overall_risk,
            "total_score": round(total_score, 2),
            "max_score": 3.0,
            "recommendation": recommendation,
            "details": results,
            "chest_reference": ctr_data['chest_bbox']
        }
    def generate_detailed_report(self, assessment: Dict) -> str:
        """
        Generate comprehensive clinical report
        """
        ctr_detail = assessment['details']['ctr']
        pos_detail = assessment['details']['position']
        vert_detail = assessment['details']['vertical']
        
        
        report_data = {
        'overall_risk': assessment['overall_risk'],
        'total_score': assessment['total_score'],
        'max_score': assessment['max_score'],
        'ctr_detail': ctr_detail,
        'pos_detail': pos_detail,
        'vert_detail': vert_detail,
        'recommendation': assessment['recommendation']
    }
        return report_data