import cv2
import os
import numpy as np
import pydicom as dicom
from ultralytics import YOLO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import load


class CombinedPipeline:
    def __init__(self):
        self.classification_model = YOLO('cls_80epoch.pt')
        self.segmentation_model = YOLO('anatomic_segmentation_0708.pt')
        self.lesion_detection_model = YOLO('lesion_detection_15epoch_297.pt')
        # self.regression_model = load('rf_full_pipeline.joblib')
        self.regression_model = load('rf_full_pipeline_240808.joblib')
        # self.regression_model = load('svm_full_pipeline.joblib')

    def process_dicom_and_predict(self, dicom_file, excel_file, output_path):
        # Process DICOM and update Excel
        self.process_dicom(dicom_file, excel_file, output_path)

        # Prepare updated data for regression
        prepared_data = self.prepare_data_for_regression(excel_file)

        # Make prediction
        prediction = self.make_prediction(prepared_data)

        return prediction

    def process_dicom(self, dicom_file, excel_file, output_path):
        best_frame = self.extract_best_frame(dicom_file)
        if best_frame is not None:
            segmentation_result = self.segment_frame(best_frame)
            detection_results = self.detect_lesions(segmentation_result, output_path)
            self.update_excel(excel_file, detection_results)

        else:
            print("No suitable high-quality frame found.")

    def extract_best_frame(self, dicom_file):
        ds = dicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array

        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array
        elif ds.PhotometricInterpretation == "YBR_FULL":
            pixel_array = np.frombuffer(ds.PixelData, dtype=np.uint8).reshape(ds.Rows, ds.Columns, 3)
        pixel_array = pixel_array.astype(np.uint8)

        best_frame = None
        best_confidence = -1

        for j in range(pixel_array.shape[0]):
            slice = pixel_array[j]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_slice = clahe.apply(slice)

            enhanced_slice_rgb = cv2.cvtColor(enhanced_slice, cv2.COLOR_GRAY2RGB)

            results = self.classification_model.predict(enhanced_slice_rgb, imgsz=512, show_labels=False,
                                                        show_boxes=False)
            confidence = results[0].probs.top1conf.item()

            if int(results[0].probs.top1) == 0 and confidence > best_confidence:
                best_frame = enhanced_slice
                best_confidence = confidence

        return best_frame

    def segment_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        results = self.segmentation_model.predict(frame_rgb, imgsz=512, conf=0.1, show_labels=False, show_boxes=False)
        segmented_frame = self.textAndContour_segment(frame, results)
        return segmented_frame

    def textAndContour_segment(self, img, results):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        check_lad_lcx = 0

        if results[0].masks is not None:
            for idx, prediction in enumerate(results[0].boxes.xywhn):
                class_id_int = int(results[0].boxes.cls[idx].item())
                poly = results[0].masks.xyn[idx].tolist()
                poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
                poly *= [w, h]

                # if class_id_int == 0 and check_lumen == 0:
                if class_id_int == 0:
                    cv2.polylines(img, [poly.astype('int')], True, (255, 0, 0), 1)
                    check_lad_lcx += 1
                elif class_id_int == 1:
                    cv2.polylines(img, [poly.astype('int')], True, (0, 255, 0), 1)
                    check_lad_lcx += 1
                elif class_id_int == 2 and check_lad_lcx == 0:
                    cv2.polylines(img, [poly.astype('int')], True, (0, 0, 255), 1)

        return img

    def save_result(self, segmented_frame, output_path):
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, 'best_segmented_frame.png'), segmented_frame)

    def detect_lesions(self, frame, output_path):
        results_detect = self.lesion_detection_model.predict(frame, imgsz=512, conf=0.01, show_labels=True)

        annotated_frame_detect = results_detect[0].plot(labels=True)
        # Save the lesion detection result
        cv2.imwrite(os.path.join(output_path, 'lesion_detection_result.png'), annotated_frame_detect)

        detected_classes = []
        for r in results_detect:
            for c in r.boxes.cls:
                detected_classes.append(self.lesion_detection_model.names[int(c)])

        return detected_classes

    def update_excel(self, excel_file, detection_results):
        df = pd.read_excel(excel_file)
        last_9_columns = df.columns[-9:].tolist()
        results_dict = {col: 0 for col in last_9_columns}

        if detection_results:
            for detected_class in detection_results:
                if detected_class in last_9_columns:
                    results_dict[detected_class] = 1

        for col, value in results_dict.items():
            df.loc[df.index[-1], col] = value

        df.to_excel(excel_file, index=False)
        print(f"Excel file '{excel_file}' has been updated with detection results.")

    def prepare_data_for_regression(self, excel_file):
        data = pd.read_excel(excel_file)

        categorical_features = ['Vessel', 'PCI_LAD', 'PCI_LCX', 'PCI_RCA', 'PA_SEX', 'YJ_SEX', 'PA_PRE_PCI',
                                'PA_PRE_CABG',
                                'PA_PRE_MI', 'PA_MI_LOC', 'PA_PRE_CHF', 'PA_CVA', 'PA_DM', 'PA_HBP', 'PA_PRE_CRF',
                                'PA_PRE_CRF_DIA', 'PA_PRE_CRF_DIA_TYPE', 'PA_SMOKING', 'YJ_SMOKING',
                                'YJ_Current_SMOKING',
                                'PA_DYSLIPID', 'PA_FHX_CAD', 'PA_DX', 'YJ_DX', 'YJ_DX_Final', 'LAD_proximal',
                                'LAD_middle',
                                'LAD_distal', 'LCX_proximal', 'LCX_middle', 'LCX_distal', 'RCA_proximal', 'RCA_middle',
                                'RCA_distal']
        continuous_features = ['PA_AGE', 'PA_HEIGHT', 'PA_WEIGHT', 'YJ_BMI', 'PA_PACK_YEAR', 'PA_QUIT_YEAR', 'WBC',
                               'Hb',
                               'Platelet', 'BUN', 'Cr', 'AST', 'ALT', 'LDL-Choleterol', 'HDL-Cholesterol', 'TG']

        expected_columns = categorical_features + continuous_features
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"Missing column in data: {col}")

        return data[expected_columns]

    def make_prediction(self, prepared_data):
        return self.regression_model.predict(prepared_data)


# Usage
pipeline = CombinedPipeline()
dicom_file = 'F213-RCA(1)_anon.dcm'
excel_file = 'FFR_Regression_Test_F213_RCA1.xlsx'
output_folder = 'INFERENCE_IMAGE'

try:
    prediction = pipeline.process_dicom_and_predict(dicom_file, excel_file, output_folder)
    print("FFR Prediction:", prediction)
except Exception as e:
    print(f"An error occurred: {str(e)}")