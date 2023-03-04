#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <vector>

//1910981 ������

using namespace std;
using namespace cv;

//������ �κ�1-������ ����
#define OUTPUT_VIDEO_NAME "out.avi"


enum Det {
	tl_x = 0,
	tl_y = 1,
	br_x = 2,
	br_y = 3,
	score = 4,
	class_idx = 5
};

struct Detection {
	cv::Rect bbox;
	float score;
	int class_idx;
};

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
	auto y = torch::zeros_like(x);
	y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
	y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
	y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
	y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
	return y;
}

void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
	const at::TensorAccessor<float, 2>& det,
	std::vector<cv::Rect>& offset_box_vec,
	std::vector<float>& score_vec)
{
	for (int i = 0; i < offset_boxes.size(0); i++) {
		offset_box_vec.emplace_back(
			cv::Rect(cv::Point(offset_boxes[i][Det::tl_x],
				offset_boxes[i][Det::tl_y]),
				cv::Point(offset_boxes[i][Det::br_x],
					offset_boxes[i][Det::br_y])));
		score_vec.emplace_back(det[i][Det::score]);
	}
}

void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
	float scale, const cv::Size& img_shape)
{
	auto clip = [](float n, float lower, float upper)
	{
		return std::max(lower, std::min(n, upper));
	};
	std::vector<Detection> detections;
	for (auto& i : data) {
		float x1 = (i.bbox.tl().x - pad_w) / scale; // x padding
		float y1 = (i.bbox.tl().y - pad_h) / scale; // y padding
		float x2 = (i.bbox.br().x - pad_w) / scale; // x padding
		float y2 = (i.bbox.br().y - pad_h) / scale; // y padding
		x1 = clip(x1, 0, img_shape.width);
		y1 = clip(y1, 0, img_shape.height);
		x2 = clip(x2, 0, img_shape.width);
		y2 = clip(y2, 0, img_shape.height);
		i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
	}
}

std::vector<float> LetterboxImage(const cv::Mat& src,
	cv::Mat& dst, const cv::Size& out_size)
{
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;
	float scale = std::min(out_w / in_w, out_h / in_h);
	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);
	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;
	cv::copyMakeBorder(dst, dst, top, down, left, right,
		cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	std::vector<float> pad_info{ static_cast<float>(left),
	static_cast<float>(top), scale };
	return pad_info;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Wrong path of weight and image" << endl;
		return -1;
	}
	float conf_thres = 0.25; // ���� ��Ȯ�� ����
	float iou_thres = 0.45; // BBox ��Ȯ�� ����
	
	//������ �κ�2-Ŭ����
	// Ŭ���� �̸�
	std::vector<std::string> class_names = {
	"10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H",
		"3S", "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S",
		"6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C",
		"8D", "8H", "8S", "9C", "9D", "9H", "9S"
	};


	// ������ġ �� ��ü
	torch::jit::script::Module model;
	try {
		// ù��° ���� (argv[1]) �� weight ���� ���
		model = torch::jit::load(argv[1]);
		cout << "YOLO model loaded" << endl;
	}
	catch (const c10::Error& e) {
		cerr << "cannot load the weight file" << endl;
		cerr << e.backtrace() << endl;
		return -1;
	}
	// �ι�° ���� (argv[2]) �� �̹��� ���� ���
	

	cv::Mat img;
	//������ �κ�3-roi����
	Mat roi, roi_1;

	//������ �ҷ�����
	VideoCapture cap(argv[2]);
	VideoWriter videoWriter;
	if (!cap.isOpened()) {
		cout << "���� ����" << endl;
		return -1;
	}

	//������ �κ�4-������ ����
	float videoFPS = cap.get(cv::CAP_PROP_FPS);
	int videoWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int videoHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	videoWriter.open(OUTPUT_VIDEO_NAME, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		videoFPS, cv::Size(videoWidth, videoHeight), true);

	if (!videoWriter.isOpened())
	{
		std::cout << "Can't write video !!! check setting" << std::endl;
		return -1;
	}

	//�����Ѻκ�5-roi
	int x = 0;
	int y = 0;

	cv::namedWindow("Result");
	while (1) {
		bool ret = cap.read(img);
		
		if (!ret) {
			cout << "���������� �б� ����(����)" << endl;
			break;
		}
		
		cv::Mat img_input = img.clone();

		//������ �κ�6-roi
		roi = img_input(Rect(Point(x, y), Point(x + 600, y + 600)));
		roi_1 = roi.clone();

		std::vector<float> pad_info = LetterboxImage(
			roi_1, roi_1, cv::Size(640, 640));
		const float pad_w = pad_info[0];
		const float pad_h = pad_info[1];
		const float scale = pad_info[2];
		

		cv::resize(roi_1, roi_1,
			cv::Size(640, 640)); // ũ�� ����
		cv::cvtColor(roi_1, roi_1,
			cv::COLOR_BGR2RGB); // BGR -> RGB
		roi_1.convertTo(roi_1, CV_32FC3,
			1.0f / 255.0f); // normalization 1/255
		
		// �ټ� ��ü ����
		auto tensor_img = torch::from_blob(roi_1.data,
			{ 1, roi_1.rows, roi_1.cols,
			roi_1.channels() }, torch::kFloat);
		// BHWC -> BCHW (Batch, Channel, Height, Width)
		tensor_img = tensor_img.permute({ 0, 3, 1, 2 }).contiguous();
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(tensor_img);
		


		// �߷�
		torch::jit::IValue output = model.forward(inputs);
		auto detections = output.toTuple()->elements()[0].toTensor();
		constexpr int item_attr_size = 5;
		int batch_size = detections.size(0); // ��ġ ũ��
		auto num_classes = detections.size(2)
			- item_attr_size; // Ŭ���� ��
		// confidence > threshold ���ǿ� �ش��ϴ� �ĺ��� ����
		auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);
		auto det = torch::masked_select(detections[0], conf_mask[0]).view(
			{ -1, num_classes + item_attr_size });
		

		// det ũ�Ⱑ 0 �̸� ����� ����� �����ϴ� �ʴ� ��
		if (0 == det.size(0)) {
			//������ �κ�7
			cout << "��ã��" << endl;
		}


		std::vector<std::vector<Detection>> result;
		result.reserve(batch_size);
		// ���ھ� ��� = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
		det.slice(1, item_attr_size, item_attr_size + num_classes)
			*= det.select(1, 4).unsqueeze(1);
		// ��ǥ ��� (center x, center y, width, height) to (x1, y1, x2, y2)
		torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));
		// Ŭ���� ���ھ ���� ���� �͸� ����
		std::tuple<torch::Tensor, torch::Tensor> max_classes =
			torch::max(det.slice(1, item_attr_size,
				item_attr_size + num_classes), 1);
		

		// class score
		auto max_conf_score = std::get<0>(max_classes);
		// index
		auto max_conf_index = std::get<1>(max_classes);
		max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
		max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);
		// shape: n * 6, top-left x/y (0,1), 
		// bottom-right x/y (2,3), score(4), 
		// class index(5)
		det = torch::cat({ box.slice(1, 0, 4),
		max_conf_score, max_conf_index }, 1);
		

		// NMS ����� �ϱ� ���� ����� �ڽ� ����� ��ġ ����ȭ
		constexpr int max_wh = 4096;
		auto c = det.slice(1, item_attr_size,
			item_attr_size + 1) * max_wh;
		auto offset_box = det.slice(1, 0, 4) + c;
		std::vector<cv::Rect> offset_box_vec;
		std::vector<float> score_vec;
		

		// copy data back to cpu
		auto offset_boxes_cpu = offset_box.cpu();
		auto det_cpu = det.cpu();
		const auto& det_cpu_array = det_cpu.accessor<float, 2>();
		// �ټ� ��ü�� Detection ��ü�� ����
		Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(),
			det_cpu_array, offset_box_vec, score_vec);
		// NMS: Non-Maximum Suppression ����
		std::vector<int> nms_indices;
		cv::dnn::NMSBoxes(offset_box_vec, score_vec,
			conf_thres, iou_thres, nms_indices);
		

		// NMS ��� ���� �ڽ����� ��ġ ��ǥ ����
		std::vector<Detection> det_vec;
		for (int index : nms_indices) {
			Detection t;
			const auto& b = det_cpu_array[index];
			t.bbox = cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
				cv::Point(b[Det::br_x], b[Det::br_y]));
			t.score = det_cpu_array[index][Det::score];
			t.class_idx = det_cpu_array[index][Det::class_idx];
			det_vec.emplace_back(t);
		}
		// ���� �̹��� ũ�⿡ ���߾� ��ǥ�� ����
		ScaleCoordinates(det_vec, pad_w, pad_h, scale, roi.size());
		result.emplace_back(det_vec);
		

		// �̹����� ���� ����� ǥ��
		for (const auto& detection : result[0]) {
			const auto& box = detection.bbox;
			float score = detection.score;
			int class_idx = detection.class_idx;
			cv::rectangle(roi, box, cv::Scalar(0, 0, 255), 2);
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << score;
			std::string s = class_names[class_idx] + " " + ss.str();
			auto font_face = cv::FONT_HERSHEY_DUPLEX;
			auto font_scale = 1.0;
			int thickness = 1;
			int baseline = 0;
			auto s_size = cv::getTextSize(s, font_face,
				font_scale, thickness, &baseline);
			cv::rectangle(roi,
				cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
				cv::Point(box.tl().x + s_size.width, box.tl().y),
				cv::Scalar(0, 0, 255), -1);
			cv::putText(roi, s, cv::Point(box.tl().x, box.tl().y - 5),
				font_face, font_scale,
				cv::Scalar(255, 255, 255), thickness);
			
			//������ �κ�8 - roi
			roi.copyTo(img(Rect(Point(x, y), Point(x + 600, y + 600))));
			

		}

		videoWriter << img;
		cv::imshow("Result", img);
		
		//������ �κ�9-roi �̵���Ű��
		if (x<600){
			x += 100;
		}
		else if (x == 600) {
			if (y == 100) {
				x = 0;
				y = 0;
			}
			else {
				x = 0;
				y += 100;
			}
		}
		
	
		//������ ���
		int fps = (int)cap.get(CAP_PROP_FPS);
		int delay = cvRound(1000 / fps);
		int key = waitKey(delay-15);
		if (key == 27) break;

	}
	cap.release();
	return 0;
}

