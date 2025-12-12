# modules/DistanceCompensator.py

import numpy as np

class DistanceCompensator:
    """
    Handles distance compensation for elevated objects
    Converts slant distance to horizontal distance using geometric optics
    """
    
    # Typical object heights database (mm)
    OBJECT_HEIGHTS = {
        "traffic light": 5000,      # 5m - standard traffic light height
        "street sign": 4000,        # 4m - street sign
        "stop sign": 2500,          # 2.5m
        "person": 1700,             # 1.7m - average adult height
        "car": 1500,                # 1.5m - sedan height
        "truck": 3000,              # 3m - truck height
        "bus": 3500,                # 3.5m
        "bicycle": 1100,            # 1.1m
        "motorcycle": 1200,         # 1.2m
    }
    
    def __init__(self, camera_params):
        """
        Initialize the compensator
        
        Args:
            camera_params: dict containing:
                - image_height: Image height (pixels) - required
                - camera_height: Camera height from ground (mm) - default 1500mm
                - fov_vertical: Vertical field of view (degrees) - optional
                - focal_length: Focal length (pixels) - optional, choose one with fov_vertical
        """
        self.image_height = camera_params['image_height']
        self.camera_height = camera_params.get('camera_height', 1500)
        
        # Calculate vertical field of view
        if 'fov_vertical' in camera_params:
            self.fov_v = np.radians(camera_params['fov_vertical'])
        elif 'focal_length' in camera_params:
            focal_length = camera_params['focal_length']
            # Calculate FOV from focal length: FOV = 2 * arctan(sensor_height / (2 * focal_length))
            # Assume sensor_height = image_height (normalized)
            self.fov_v = 2 * np.arctan(self.image_height / (2 * focal_length))
        else:
            # Default value: phone cameras typically 60-70 degrees
            print("Warning: No FOV or focal_length provided, using default 65 degrees")
            self.fov_v = np.radians(65)
        
        # Angle per pixel
        self.angle_per_pixel = self.fov_v / self.image_height
        
        print(f"DistanceCompensator initialized:")
        print(f"  - Image height: {self.image_height}px")
        print(f"  - Camera height: {self.camera_height}mm")
        print(f"  - Vertical FOV: {np.degrees(self.fov_v):.2f}°")
        print(f"  - Angle per pixel: {np.degrees(self.angle_per_pixel):.4f}°")
    
    def pixel_to_pitch_angle(self, y_pixel):
        """
        Convert pixel y coordinate to pitch angle
        
        Args:
            y_pixel: Object's y coordinate in image (bbox center)
        
        Returns:
            pitch_angle: Pitch angle (radians)
                        Positive = camera looking up (object above sight line)
                        Negative = camera looking down (object below sight line)
        """
        # Image coordinate system: (0,0) at top-left, y increases downward
        # Convert to visual coordinate system: image center as origin, upward is positive
        y_center = self.image_height / 2
        pixel_offset = y_center - y_pixel  # Positive = object is above
        
        # Pixel offset → angle offset
        pitch_angle = pixel_offset * self.angle_per_pixel
        
        return pitch_angle
    
    def get_object_height(self, object_type):
        """
        Get typical height of object
        
        Args:
            object_type: Object type string
        
        Returns:
            height: Object height (mm)
        """
        return self.OBJECT_HEIGHTS.get(
            object_type.lower(), 
            self.camera_height  # Default: same height as camera
        )
    
    def compensate_distance(self, slant_distance, y_pixel, object_type):
        """
        Compensate for height difference to calculate horizontal distance
        
        Core principle:
        1. Calculate pitch angle θ based on object position in image
        2. Calculate height difference Δh = object_height - camera_height
        3. Use trigonometric relationship to decompose distance:
           - tan(θ) = Δh / horizontal_distance
           - horizontal_distance = Δh / tan(θ)
        
        Args:
            slant_distance: Original distance from MiDaS calculation (mm)
            y_pixel: Object bbox center y coordinate
            object_type: Object type
        
        Returns:
            compensated_distance: Compensated horizontal distance (mm)
        """
        # 1. Get object height
        object_height = self.get_object_height(object_type)
        height_diff = object_height - self.camera_height
        
        # 2. Calculate pitch angle
        pitch = self.pixel_to_pitch_angle(y_pixel)
        
        # 3. Distance compensation strategy
        if abs(pitch) < np.radians(3):  # < 3 degrees, nearly horizontal
            # Height difference has minimal impact, return original distance
            return slant_distance
        
        elif abs(height_diff) < 500:  # Height difference < 0.5m
            # Object nearly at same height as camera (e.g., pedestrian, car)
            return slant_distance
        
        else:
            # Compensation needed (e.g., traffic light)
            
            # Method A: Geometric calculation based on pitch angle
            if abs(pitch) > np.radians(5):  # Angle large enough, reliable
                # horizontal_distance = |Δh / tan(pitch)|
                horizontal_distance_geometric = abs(height_diff / np.tan(pitch))
                
                # Limit to reasonable range (avoid outliers)
                horizontal_distance_geometric = np.clip(
                    horizontal_distance_geometric, 
                    slant_distance * 0.5,  # Minimum 50% of slant distance
                    slant_distance * 2.0   # Maximum 200% of slant distance
                )
            else:
                horizontal_distance_geometric = slant_distance
            
            # Method B: Vector decomposition from slant distance
            # slant² = horizontal² + vertical²
            # Assumption: vertical ≈ Δh (approximation, as slant is not strict Euclidean distance)
            vertical_component_sq = height_diff ** 2
            horizontal_distance_vector = np.sqrt(
                max(0, slant_distance**2 - vertical_component_sq)
            )
            
            # Hybrid strategy: larger angle, more trust in geometric method
            angle_weight = min(1.0, abs(pitch) / np.radians(30))  # Weight=1 at 30 degrees
            
            compensated_distance = (
                angle_weight * horizontal_distance_geometric + 
                (1 - angle_weight) * horizontal_distance_vector
            )
            
            return compensated_distance
    
    def batch_compensate(self, detections, y_pixels, object_types):
        """
        Batch process multiple detections
        
        Args:
            detections: list of distances (mm)
            y_pixels: list of y coordinates
            object_types: list of object type strings
        
        Returns:
            compensated_distances: list of compensated distances
        """
        return [
            self.compensate_distance(dist, y, obj_type)
            for dist, y, obj_type in zip(detections, y_pixels, object_types)
        ]


def calibrate_camera_focal_length(known_distance_mm, known_height_mm, 
                                   pixel_height, image_height):
    """
    Calibrate camera focal length using a known object
    
    Usage scenario:
    1. Find an object with known height (e.g., person with 1.7m height)
    2. Stand at known distance (e.g., 5m)
    3. Take photo, measure how many pixels the object occupies
    4. Call this function to calculate focal length
    
    Args:
        known_distance_mm: Actual distance from object to camera (mm)
        known_height_mm: Actual object height (mm)
        pixel_height: Object's pixel height in image
        image_height: Total image height (pixels)
    
    Returns:
        focal_length: Focal length (pixels)
    
    Example:
        # A 1.7m tall person stands 5m away, occupies 340 pixels in image
        f = calibrate_camera_focal_length(5000, 1700, 340, 720)
        # f ≈ 1000 pixels
    """
    # Based on similar triangles:
    # pixel_height / image_height = real_height / (distance × tan(FOV/2))
    # Simplified formula: focal_length = (pixel_height × distance) / real_height
    focal_length = (pixel_height * known_distance_mm) / known_height_mm
    
    print(f"Calibration result:")
    print(f"  - Focal length: {focal_length:.2f} pixels")
    print(f"  - Vertical FOV: {np.degrees(2 * np.arctan(image_height / (2 * focal_length))):.2f}°")
    
    return focal_length


# Test code
if __name__ == "__main__":
    # Example: Camera calibration
    print("=== Camera Calibration ===")
    focal_length = calibrate_camera_focal_length(
        known_distance_mm=5000,   # 5 meters
        known_height_mm=1700,     # 1.7m tall person
        pixel_height=340,         # Occupies 340 pixels in image
        image_height=720          # 720p image height
    )
    
    print("\n=== Distance Compensation Test ===")
    # Initialize compensator
    compensator = DistanceCompensator({
        'image_height': 720,
        'camera_height': 1500,
        'focal_length': focal_length
    })
    
    # Test scenario: Detected a traffic light
    slant_distance = 45400  # MiDaS calculated 45.4m
    y_pixel = 150           # Upper part of image (y=150, image height 720)
    
    compensated = compensator.compensate_distance(
        slant_distance, y_pixel, "traffic light"
    )
    
    print(f"\nTraffic Light Detection:")
    print(f"  - Original distance: {slant_distance/1000:.1f}m")
    print(f"  - Y position: {y_pixel}px (upper part of image)")
    print(f"  - Compensated distance: {compensated/1000:.1f}m")
    print(f"  - Correction: {(compensated/slant_distance - 1)*100:.1f}%")