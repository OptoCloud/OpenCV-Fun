#include <iostream>

#include <opencv2/opencv.hpp>
#include <optional>
#include <numbers>

const float RadToDeg = 180.f / std::numbers::pi_v<float>;
const float DegToRad = std::numbers::pi_v<float> / 180.f;

cv::Point RectMiddle(const cv::Rect& rect)
{
	return cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier& GetFaceCC()
{
	if (face_cascade.empty())
	{
		face_cascade.load("haarcascade_frontalface_alt.xml");
	}
	return face_cascade;
}

cv::CascadeClassifier eye_cascade;
cv::CascadeClassifier& GetEyeCC()
{
	if (eye_cascade.empty())
	{
		eye_cascade.load("haarcascade_eye.xml");
	}
	return eye_cascade;
}

bool TryGetEyes(const cv::Mat& image, const cv::Point& offset, cv::Point& leftEye, cv::Point& rightEye)
{
	if (image.empty())
	{
		return false;
	}

	// Detect eyes
	std::vector<cv::Rect> detections;
	GetEyeCC().detectMultiScale(image, detections, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	
	if (detections.size() < 2)
	{
		return false;
	}

	std::sort(detections.begin(), detections.end(), [](const cv::Rect& a, const cv::Rect& b) { return a.area() > b.area(); });

	if (detections[0].empty() || detections[1].empty())
	{
		return false;
	}
	
	if (detections[0].x < detections[1].x)
	{
		leftEye = RectMiddle(detections[0]);
		rightEye = RectMiddle(detections[1]);
	}
	else
	{
		leftEye = RectMiddle(detections[1]);
		rightEye = RectMiddle(detections[0]);
	}

	// Offset the eyes to the face's position
	leftEye.x += offset.x;
	leftEye.y += offset.y;
	rightEye.x += offset.x;
	rightEye.y += offset.y;

	return true;
}

struct Face
{
	cv::Point pos;
	float width;
	float tiltRads;

	cv::Point leftEye;
	cv::Point rightEye;
	
};
void GetFaces(const cv::Mat& image, std::vector<Face>& faces)
{
	faces = std::vector<Face>{};
	
	// Check if image is larger than 1x1
	if (image.empty())
	{
		return;
	}

	// Detect faces
	std::vector<cv::Rect> detections;
	GetFaceCC().detectMultiScale(image, detections, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	
	for (const cv::Rect& detection : detections)
	{
		if (detection.empty()) continue;

		Face face{
			.pos = RectMiddle(detection)
		};

		if (!TryGetEyes(image(detection), cv::Point(detection.x, detection.y), face.leftEye, face.rightEye)) continue;

		// Attempt to guess the orientation (tilt) of the face
		// This is done by calculating the angle between the eyes
		if (face.leftEye.x < face.rightEye.x)
		{
			face.tiltRads = std::atan2(face.rightEye.y - face.leftEye.y, face.rightEye.x - face.leftEye.x);
		}
		else
		{
			face.tiltRads = std::atan2(face.leftEye.y - face.rightEye.y, face.leftEye.x - face.rightEye.x);
		}

		// Calculate the width of the face based on the detection which is a square and the tilt of the head
		face.width = (detection.width / std::cos(face.tiltRads)) * 0.5f;
		
		faces.push_back(face);
	}
}

int main()
{
	cv::setNumThreads(10);

	// Open webcam
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the web cam" << std::endl;
		return -1;
	}

	// Get a cascade classifier for detecting eyes
	cv::CascadeClassifier eye_cascade;
	if (!eye_cascade.load("haarcascade_eye.xml"))
	{
		std::cout << "Cannot load the eye cascade classifier" << std::endl;
		return -1;
	}

	// Get a cascade classifier for detecting faces
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load("haarcascade_frontalface_alt.xml"))
	{
		std::cout << "Cannot load the face cascade classifier" << std::endl;
		return -1;
	}

	// Allocate window
	cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
	
	while (1)
	{
		cv::Mat frame;

		// Capture frame-by-frame
		cap >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
			break;
		
		// Convert to grayscale
		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		
		// Detect faces
		std::vector<Face> faces;
		GetFaces(gray, faces);

		for (const Face& face : faces)
		{
			// Draw a line between the eyes
			cv::line(frame, face.leftEye, face.rightEye, cv::Scalar(0, 255, 0), 2);
			
			// Draw a line indicating the tilt of the face
			cv::line(frame, face.pos, face.pos + cv::Point(std::cos(face.tiltRads) * face.width, std::sin(face.tiltRads) * face.width), cv::Scalar(0, 0, 255), 2);
			
			// Draw a ellipse approximating the face
			cv::ellipse(frame, face.pos, cv::Size(face.width, face.width * 1.5f), face.tiltRads * RadToDeg, 0, 360, cv::Scalar(255, 0, 0), 2);

			// Draw two eclipses approximating the eyes,and rotate them
			cv::ellipse(frame, face.leftEye, cv::Size(20, 10), face.tiltRads * RadToDeg, 0, 360, cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::ellipse(frame, face.rightEye, cv::Size(20, 10), face.tiltRads * RadToDeg, 0, 360, cv::Scalar(0, 0, 255), 2, 8, 0);
		}		
		
		std::string debugText;
		switch (faces.size())
		{
		case 0:
			debugText = "No face detected";
			break;
		case 1:
			debugText = "1 face detected";
			break;
		default:
			debugText = std::to_string(faces.size()) + " faces detected";
			break;
		}

		cv::putText(frame, debugText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		if (!faces.empty())
		{
			cv::putText(frame, "Tilt: " + std::to_string(faces[0].tiltRads * RadToDeg), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		}

		// Show frame
		cv::imshow("Webcam", frame);

		// Press Q or ESC to quit
		if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q')
			break;
	}

	return 0;
}