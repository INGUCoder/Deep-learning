#include "frame_state_detector.hpp"
#include <math.h>
#include <mutex>
#include <cnrt.h>
#include <thread>
#include <semaphore.h>


#ifdef WIN32
#include <io.h>
#endif
//using namespace base::CLock;
#include <algorithm>

namespace facethink {
    namespace detfacemtcnn {
        using namespace std; 
        void DataWrite_CMode(std::string filename, vector<int> buf, int len)
        {
            ofstream outfile(filename, ios::out); 
            string temp; 
            for(int i = 0; i < len; i++) 
            { 
                outfile << buf[i] << std::endl; 
            } 
            outfile.close();
        }

        void DataWrite_CMode(std::string filename, vector<float> buf, int len)
        {
            ofstream outfile(filename, ios::out); 
            string temp; 
            for(int i = 0; i < len; i++) 
            { 
                outfile << buf[i] << std::endl; 
            } 
            outfile.close();
        }

        void DataWrite_CMode(std::string filename, int *buf, int len)
        {
            ofstream outfile(filename, ios::out); 
            string temp; 
            for(int i = 0; i < len; i++) 
            { 
                outfile << buf[i] << std::endl; 
            } 
            outfile.close();
        }

        using namespace cambricon;
        CCriticalSection FrameStateDetector::m_criticalsection;
        FrameStateDetector::FrameStateDetector(
                const std::vector<std::string>& det1_model_binarys,
		const std::string& det2_model_binary;
		const std::string& det3_model_binary;
                const std::string& config_file) {

	    P_model_binary_ = det1_model_binary;

            CambriconUtil* cam2_ = new CambriconUtil();
            CambriconUtil* cam3_ = new CambriconUtil();
            cam2_->my_init_net(det2_model_prototxt, det2_model_binary, 0);
            cam3_->my_init_net(det3_model_prototxt, det3_model_binary, 0);
            _myCams_.push_back(cam2_);
            _myCams_.push_back(cam2_);
            _myCams_.push_back(cam3_);

            CambriconUtil* cam1 = new CambriconUtil();
            cam1->init_net(det1_model_prototxt, det1_model_binary);
            lcams_.push_back(cam1);
            CambriconUtil* cam2 = new CambriconUtil();
            cam2->init_net(det2_model_prototxt, det2_model_binary);
            lcams_.push_back(cam2);
            CambriconUtil* cam3 = new CambriconUtil();
            cam3->init_net(det3_model_prototxt, det3_model_binary);
            lcams_.push_back(cam3);

            for (int i = 0; i < det1_model_prototxts.size(); ++i)
            {
                CambriconUtil* cam = new CambriconUtil();
                cam->init_net(det1_model_prototxts[i], det1_model_binarys[i]);
                cams_net_1_s_.push_back(cam);
                binary_inputs_.push_back(det1_model_prototxts[i]);
                binary_inputs_.push_back(det1_model_binarys[i]);
            }

            initialPnet();

            
            for (size_t i = 0; i < 3; i++)
            {
                cv::Size input_geometry;
                int num_channel = lcams_[i]->input_shape[0].c;
                num_channel = lcams_[i]->input_shape[0].c;
                //input_geometry = cv::Size(input_layer->width(), input_layer->height());
                //input_geometry = cv::Size(input_layer->width(), input_layer->height());
                input_geometry = cv::Size(lcams_[i]->input_shape[0].w, lcams_[i]->input_shape[0].h);
                //else input_geometry = cv::Size(12, 12);
                input_geometry_.push_back(input_geometry);
                std::cout << "input_geometry " << i << lcams_[i]->input_shape[0].w << lcams_[i]->input_shape[0].h << std::endl;
                if (i == 0)
                    num_channels_ = num_channel;
                else if (num_channels_ != num_channel)
                    std::cout << "Error: The number channels of the nets are different!" << std::endl;
            }
#ifdef WIN32
            if (_access(config_file.c_str(), 0) != -1) {
                config_.ReadIniFile(config_file);
            }
#else
            if (access(config_file.c_str(), 0) != -1) {
                config_.ReadIniFile(config_file);
            }
#endif
        }

        FrameStateDetector::FrameStateDetector(
                const std::string& det1_model_file,
                const std::string& det2_model_file,
                const std::string& det3_model_file,
                const std::string& config_file) {

#ifdef WIN32
            if (_access(config_file.c_str(), 0) != -1) {
                config_.ReadIniFile(config_file);
            }
#else
            if (access(config_file.c_str(), 0) != -1) {
                config_.ReadIniFile(config_file);
            }
#endif
            /*facethinkcaffe::NetBuilder<float> net_builder;
              nets_[0] = net_builder.Create(det1_model_file);
              nets_[1] = net_builder.Create(det2_model_file);
              nets_[2] = net_builder.Create(det3_model_file);

              for (size_t i = 0; i < 3; i++)
              {
              cv::Size input_geometry;
              int num_channel;

              auto input_layer = nets_[i]->blob("data");
              num_channel = input_layer->channels();
              input_geometry = cv::Size(input_layer->width(), input_layer->height());
              input_geometry_.push_back(input_geometry);

              if (i == 0)
              num_channels_ = num_channel;
              else if (num_channels_ != num_channel)
              std::cout << "Error: The number channels of the nets are different!" << std::endl;
              }*/
        }

        void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
        {
            net_time_count = 0;
            Preprocess(img);
            P_Net();
            R_Net();
            O_Net();
            vector<float>NetCost;
            NetCost.push_back(net_time_count);

            for (auto &bounding_box : boxes_)
            {
                rectangles.push_back(
                        cv::Rect(bounding_box[0] * 1.0 / config_.SCALE,
                            bounding_box[1] * 1.0 / config_.SCALE,
                            (bounding_box[2] * 1.0 - bounding_box[0] * 1.0) / config_.SCALE,
                            (bounding_box[3] * 1.0 - bounding_box[1] * 1.0) / config_.SCALE));
            }
        }

        void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, float scale)
        {
            net_time_count = 0;
            Preprocess(img);
            P_Net();
            R_Net();
            O_Net();
            rectangles.clear();
            for (auto &bounding_box : boxes_)
            {
                rectangles.push_back(
                        cv::Rect(bounding_box[0] * 1.0 / config_.SCALE,
                            bounding_box[1] * 1.0 / config_.SCALE,
                            (bounding_box[2] * 1.0 - bounding_box[0] * 1.0) / config_.SCALE,
                            (bounding_box[3] * 1.0 - bounding_box[1] * 1.0) / config_.SCALE));
            }
        }

        void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
        {
            detection(img, rectangles);

            confidence = confidence_;
        }

        void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment)
        {
            detection(img, rectangles, confidence);

            alignment.clear();
            for (auto &i : alignment_)
            {
                std::vector<cv::Point> temp_alignment;
                for (size_t j = 0; j < i.size() / 2; j++)
                {
                    temp_alignment.push_back(cv::Point(i[2 * j] * 1.0 / config_.SCALE, i[2 * j + 1] * 1.0 / config_.SCALE));
                }
                alignment.push_back(std::move(temp_alignment));
            }
        }

        void FrameStateDetector::detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment)
        {
            detection_SCALE(img, rectangles, confidence);

            alignment.clear();
            for (auto &i : alignment_)
            {
                std::vector<cv::Point> temp_alignment;
                for (size_t j = 0; j < i.size() / 2; j++)
                {
                    temp_alignment.push_back(cv::Point(i[2 * j] * 1.0 / config_.SCALE, i[2 * j + 1] * 1.0 / config_.SCALE));
                }
                alignment.push_back(std::move(temp_alignment));
            }

        }

        void FrameStateDetector::detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
        {
            Lock lock(&m_criticalsection);
            if (img.empty() || img.channels() != 3) {
                BOOST_LOG_TRIVIAL(fatal) << "Input image must has 3 channels.";
            }

            img_resized_.clear();
            scale_.clear();
            params_.clear();
            regression_box_temp_.clear();
            confidence_.clear();
            confidence_temp_.clear();
            alignment_.clear();
            alignment_temp_.clear();
            boxes_.clear();

            BOOST_LOG_TRIVIAL(debug) << "Det Face MTCNN Net: Start Detect.";
            auto time_start = std::chrono::steady_clock::now();
            //scale image
            cv::Mat scaled_img;
            cv::Size size = cv::Size((int)(img.cols * config_.SCALE), (int)(img.rows * config_.SCALE));
            cv::resize(img, scaled_img, size);
            detection(scaled_img, rectangles, 1.0 / config_.SCALE);
            confidence = confidence_;
            auto time_end = std::chrono::steady_clock::now();
            BOOST_LOG_TRIVIAL(info) << "Det Face MTCNN Net Cost Time: " << net_time_count << "ms";
            BOOST_LOG_TRIVIAL(info) << "Det Face MTCNN Total Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms";

        }

        void FrameStateDetector::Preprocess(const cv::Mat &img)
        {
            /* Convert the input image to the input image format of the network. */
            cv::Mat sample;
            if (img.channels() == 3 && num_channels_ == 1)
                cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
            else if (img.channels() == 4 && num_channels_ == 1)
                cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
            else if (img.channels() == 4 && num_channels_ == 3)
                cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
            else if (img.channels() == 1 && num_channels_ == 3)
                cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
            else
                sample = img;

            cv::Mat sample_float;
            if (num_channels_ == 3)
                sample.convertTo(sample_float, CV_32FC3);
            else
                sample.convertTo(sample_float, CV_32FC1);

            cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2RGB);
            //sample_float = sample_float;//.t(); //TODO

            img_ = sample_float;
        }

        void FrameStateDetector::initialPnet()
        {

            pnet_batch_size = 1;
            pnet_type = CNRT_FUNC_TYPE_BLOCK;
            dim = {1, 1, 1}; 

            CNRT_CHECK(cnrtInit(0)); 

            threads_.resize(mparallel_);
            for(int i = 0; i < mparallel_; i++)
            {
                sem_init(&sem_in_[i], 0, 0);
                sem_init(&sem_out_[i], 0, 0);
                CNRT_CHECK(cnrtCreateStream(&pnet_stream[i]));
                threads_[i] = new thread(&FrameStateDetector::doPnet, this, i);
            }

            for(int i = 0; i < mparallel; i++)
            {
                cnrtModel_t cnrt_model;
                cnrtFunction_t function;
                cnrtDataDescArray_t inputDescS;
                cnrtDataDescArray_t outputDescS;
                vector<int> in_shape;
                vector<int> cls_shape;
                vector<int> box_shape;
                vector<int> in_size;
                vector<int> out_size;
                vector<float*> input_cpu;
                vector<float*> output_cpu;
                void** param;
                void** inputCpuPtrS;
                void** outputCpuPtrS;
                void** inputMluPtrS;
                void** outputMluPtrS;
                int inputNum;
                int outputNum;
                unsigned int n, c, h, w;


                CNRT_CHECK(cnrtLoadModel(&cnrt_model, pnet_model_path[i].c_str()));
                CNRT_CHECK(cnrtCreateFunction(&function));
                CNRT_CHECK(cnrtExtractFunction(&function, cnrt_model, "subnet0"));

                CNRT_CHECK(cnrtInitFunctionMemory(function, pnet_type));

                CNRT_CHECK(cnrtGetInputDataDesc(&inputDescS, &inputNum , function));
                CNRT_CHECK(cnrtGetOutputDataDesc(&outputDescS, &outputNum, function));

                CNRT_CHECK(cnrtGetDataShape(inputDescS[0], &n, &c, &h, &w));
                n *= pnet_batch_size;
                in_shape.push_back(n);
                in_shape.push_back(c);
                in_shape.push_back(h);
                in_shape.push_back(w);
                in_size.push_back(n * c * h * w);
                //printf("input:%d %d %d %d\n",n,c,h,w);

                CNRT_CHECK(cnrtGetDataShape(outputDescS[0], &n, &c, &h, &w));
                n *= pnet_batch_size;
                box_shape.push_back(n);
                box_shape.push_back(c);
                box_shape.push_back(h);
                box_shape.push_back(w);
                out_size.push_back(n * c * h * w);

                CNRT_CHECK(cnrtGetDataShape(outputDescS[1], &n, &c, &h, &w));
                n *= pnet_batch_size;
                cls_shape.push_back(n);
                cls_shape.push_back(c);
                cls_shape.push_back(h);
                cls_shape.push_back(w);
                out_size.push_back(n * c * h * w);

                inputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * inputNum));
                for (int j = 0; j < inputNum; j++)
                {
                    int ip;
                    float* input_data;
                    cnrtDataDesc_t inputDesc = inputDescS[j];
                    if(m_conv_first)
                        CNRT_CHECK(cnrtSetHostDataLayout(inputDesc, CNRT_UINT8, CNRT_NCHW));
                    else
                        CNRT_CHECK(cnrtSetHostDataLayout(inputDesc, CNRT_FLOAT32, CNRT_NCHW));
                    CNRT_CHECK(cnrtGetHostDataCount(inputDesc, &ip));
                    //  CNRT_CHECK(cnrtCloseReshapeOfOneDataDesc(inputDesc));
                    ip *= pnet_batch_size;
                    input_data = reinterpret_cast<float*>(malloc(sizeof(float) * ip));
                    input_cpu.push_back(input_data);
                    inputCpuPtrS[j] = reinterpret_cast<void*>(input_data);
                }
                outputCpuPtrS = reinterpret_cast<void**>(malloc(sizeof(void*) * outputNum));
                for (int j = 0; j < outputNum; j++)
                {
                    int op;
                    float* output_data;
                    cnrtDataDesc_t outputDesc = outputDescS[j];
                    CNRT_CHECK(cnrtSetHostDataLayout(outputDesc, CNRT_FLOAT32, CNRT_NCHW));
                    CNRT_CHECK(cnrtGetHostDataCount(outputDesc, &op));
                    op *= pnet_batch_size;
                    output_data = reinterpret_cast<float*>(malloc(sizeof(float) * op));
                    output_cpu.push_back(output_data);
                    outputCpuPtrS[j] = reinterpret_cast<void*>(output_data);
                }

                param = reinterpret_cast<void**>(malloc(sizeof(void*) * (inputNum + outputNum)));

                CNRT_CHECK(cnrtMallocBatchByDescArray(&inputMluPtrS, inputDescS, inputNum, pnet_batch_size));
                for (int j = 0; j < inputNum; j++)
                {
                    param[j] = inputMluPtrS[j];
                }

                CNRT_CHECK(cnrtMallocBatchByDescArray(&outputMluPtrS, outputDescS, outputNum, pnet_batch_size));
                for (int j = 0; j < outputNum; j++)
                {
                    param[inputNum + j] = outputMluPtrS[j];
                }

                pnet_cnrt_model.push_back(cnrt_model);
                pnet_function.push_back(function);
                pnet_inputDescS.push_back(inputDescS);
                pnet_outputDescS.push_back(outputDescS);
                pnet_in_shape.push_back(in_shape);
                pnet_cls_shape.push_back(cls_shape);
                pnet_box_shape.push_back(box_shape);
                pnet_in_size.push_back(in_size);
                pnet_out_size.push_back(out_size);
                pnet_input_cpu.push_back(input_cpu);
                pnet_output_cpu.push_back(output_cpu);
                pnet_param.push_back(param);
                pnet_inputCpuPtrS.push_back(inputCpuPtrS);
                pnet_outputCpuPtrS.push_back(outputCpuPtrS);
                pnet_inputMluPtrS.push_back(inputMluPtrS);
                pnet_outputMluPtrS.push_back(outputMluPtrS);
                pnet_inputNum.push_back(inputNum);
                pnet_outputNum.push_back(outputNum);
            }
            
        }

        void FrameStateDetector::wrapInputLayer(vector<int> input_shape, float* input_dataBuffer, vector< cv::Mat >* input_channels)
        {

            if(m_conv_first)
            {
                int width = input_shape[3];
                int height = input_shape[2];
                unsigned char* input_data = (unsigned char*)input_dataBuffer;
                int margin = width * height;
                int type=CV_8UC1;
                for(int j = 0; j < input_shape[0]; j++)
                {
                    for(int i = 0; i < input_shape[1]; i++)
                    {
                        cv::Mat channel(height, width, type, input_data);
                        input_channels->push_back(channel);
                        input_data += margin;
                    }
                }
            }
            else
            {
                int width = input_shape[3];
                int height = input_shape[2];
                float* input_data = input_dataBuffer;
                int margin = width * height;
                int type=CV_32FC1;
                for(int j = 0; j < input_shape[0]; j++)
                {
                    for(int i = 0; i < input_shape[1]; i++)
                    {
                        cv::Mat channel(height, width, type, input_data);
                        input_channels->push_back(channel);
                        input_data += margin;
                    }
                }
            }

        }

        void FrameStateDetector::pyrDown(const vector<cv::Mat>& img, vector< cv::Mat >* input_channels)
        {

            assert(img.size() == input_channels->size());
            int hs = (*input_channels)[0].rows;
            int ws = (*input_channels)[0].cols;
            cv::Mat img_resized;
            for(int i = 0; i < img.size(); i ++)
            {
                cv::resize(img[i], (*input_channels)[i], cv::Size(ws, hs),cv::INTER_NEAREST);
            }

        }

        void FrameStateDetector::doPnet(int idx)
        {

            vector<cv::Mat> pyr_channels;
            vector<BoundingBox> filterOutBoxes;
            vector<BoundingBox> nmsOutBoxes;

            unsigned dev_num;
            CNRT_CHECK(cnrtGetDeviceCount(&dev_num));
            if (dev_num == 0) return;
            cnrtDev_t dev; 
            CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0)); 
            CNRT_CHECK(cnrtSetCurrentDevice(dev));
            CNRT_CHECK(cnrtSetCurrentChannel((cnrtChannelType_t)0));

            while(1)
            {    
                sem_wait(&sem_in_[idx]);

                float cur_sc;
                pyr_channels.clear();
                wrapInputLayer(pnet_in_shape[idx], pnet_input_cpu[idx][0], &pyr_channels);
                pyrDown(sample_norm_channels, &pyr_channels);

                CNRT_CHECK(cnrtMemcpyBatchByDescArray(pnet_inputMluPtrS[idx], 
                            pnet_inputCpuPtrS[idx], 
                            pnet_inputDescS[idx],
                            pnet_inputNum[idx],
                            pnet_batch_size, 
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

                auto time_start = std::chrono::steady_clock::now();
                CNRT_CHECK(cnrtInvokeFunction(pnet_function[idx], dim, pnet_param[idx], pnet_type, pnet_stream[idx], NULLPTR));
                CNRT_CHECK(cnrtSyncStream( pnet_stream[idx]));
                auto time_end = std::chrono::steady_clock::now();

                CNRT_CHECK(cnrtMemcpyBatchByDescArray(pnet_outputCpuPtrS[idx], pnet_outputMluPtrS[idx],
                            pnet_outputDescS[idx],
                            pnet_outputNum[idx],
                            pnet_batch_size,
                            CNRT_MEM_TRANS_DIR_DEV2HOST));

                /* copy the output layer to a vector*/
                //getPnetSoftmax(pnet_output_cpu[idx][1], 2, pnet_out_size[idx][1] / 2);
                const float* begin1 = pnet_output_cpu[idx][1];
                const float* end1 = pnet_out_size[idx][1] + begin1;
                vector<float> pnet_cls(begin1, end1);

                const float* begin0 = pnet_output_cpu[idx][0];
                const float* end0 = pnet_out_size[idx][0] + begin0;
                vector<float> pnet_regs(begin0, end0);
                filterOutBoxes.clear();
                nmsOutBoxes.clear();
                cur_sc = 1.0 * pnet_in_shape[idx][3] / img_W;

                generateBoundingBox(pnet_regs, pnet_box_shape[idx], pnet_cls, pnet_cls_shape[idx], cur_sc, P_thres, filterOutBoxes);
                nms(filterOutBoxes, 0.5, UNION, nmsOutBoxes);

                mtx.lock();
                if(nmsOutBoxes.size() > 0)
                    thread_box_.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
                mtx.unlock();

                sem_post(&sem_out_[idx]);
            }

        }

        void FrameStateDetector::nms(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
        {

                filterOutBoxes.clear();
            if(boxes.size() == 0)
                return;

            //descending sort
            sort(boxes.begin(), boxes.end(), CmpBoundingBox() );
            vector<size_t> idx(boxes.size());
            for(int i = 0; i < idx.size(); i++)
            {
                idx[i] = i;
            }
            while(idx.size() > 0)
            {
                int good_idx = idx[0];
                filterOutBoxes.push_back(boxes[good_idx]);
                //hypothesis : the closer the scores are similar
                vector<size_t> tmp = idx;
                idx.clear();
                for(int i = 1; i < tmp.size(); i++)
                {
                    int tmp_i = tmp[i];
                    float inter_x1 = max( boxes[good_idx].x1, boxes[tmp_i].x1 );
                    float inter_y1 = max( boxes[good_idx].y1, boxes[tmp_i].y1 );
                    float inter_x2 = min( boxes[good_idx].x2, boxes[tmp_i].x2 );
                    float inter_y2 = min( boxes[good_idx].y2, boxes[tmp_i].y2 );

                    float w = max((inter_x2 - inter_x1 + 1), 0.0F);
                    float h = max((inter_y2 - inter_y1 + 1), 0.0F);

                    float inter_area = w * h;
                    float area_1 = (boxes[good_idx].x2 - boxes[good_idx].x1 + 1) * (boxes[good_idx].y2 - boxes[good_idx].y1 + 1);
                    float area_2 = (boxes[i].x2 - boxes[i].x1 + 1) * (boxes[i].y2 - boxes[i].y1 + 1);
                    float o = (type == UNION ? (inter_area / (area_1 + area_2 - inter_area)) : (inter_area / min(area_1 , area_2)));
                    if (o <= threshold)
                        idx.push_back(tmp_i);
                }
            }
        }

        void FrameStateDetector::generateBoundingBox(const vector<float>& boxRegs, const vector<int>& box_shape,
                const vector<float>& cls, const vector<int>& cls_shape,
                float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                )
        {

                //clear output element
                filterOutBoxes.clear();
            int stride = 2;
            int cellsize = 12;
            assert(box_shape.size() == cls_shape.size());
            assert(box_shape[3] == cls_shape[3] && box_shape[2] == cls_shape[2]);
            assert(box_shape[0] == 1 && cls_shape[0] == 1);
            assert(box_shape[1] == 4 && cls_shape[1] == 2);
            int w = box_shape[3];
            int h = box_shape[2];
            //int n = box_shape[0];
            for(int y = 0; y < h; y ++)
            {
                for(int x = 0; x < w; x ++)
                {
                    float score = cls[0 * 2 * w * h + 1 * w * h + w * y + x];
                    if (score >= threshold)
                    {
                        BoundingBox box;
                        box.dx1 = boxRegs[0 * w * h + w * y + x];
                        box.dy1 = boxRegs[1 * w * h + w * y + x];
                        box.dx2 = boxRegs[2 * w * h + w * y + x];
                        box.dy2 = boxRegs[3 * w * h + w * y + x];

                        box.x1 = floor( (stride * x + 1) / scale );
                        box.y1 = floor( (stride * y + 1) / scale );
                        box.x2 = floor( (stride * x + cellsize) / scale );
                        box.y2 = floor( (stride * y + cellsize) / scale );
                        box.score = score;

                        //add elements
                        filterOutBoxes.push_back(box);
                    }
                }
            }

        }

        void FrameStateDetector::myThreadPnet(int count1)
        {
            std::vector<int>seq;

            CambriconUtil* camPnet = new CambriconUtil();
            camPnet->init_net(binary_inputs_[2 * count1], binary_inputs_[2 * count1 + 1]);

            while(1){
                sem_wait(&sem_in_[count1]);
                cv::Mat img_resized = img_resized_[count1];
                cout << count1 << endl;

                if (img_resized.cols * img_resized.rows == 0)
                {
                    BOOST_LOG_TRIVIAL(error) << "Input image is empty.";
                    return;
                }

                cv::Mat cls_blob_img = blobFromImage(img_resized.t(), 1.0f, false);
                std::vector<float*> output_datas;
                std::vector<MShape> output_shapes;
                int time_count = 0;
                auto time_start = std::chrono::steady_clock::now();
                camPnet->run(cls_blob_img.ptr<float>(0), output_datas, output_shapes);
                auto time_end = std::chrono::steady_clock::now();
                mtx_.lock();
                timePair.push_back(make_pair(time_start, time_end));
                mtx_.unlock();
                //net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();

                BOOST_LOG_TRIVIAL(debug) << "Forward P-net: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count() << " ms";
                std::cout<<"P_Net---input_shapes[0] "
                    <<cams_net_1_s_[count1]->input_shape->n << " "
                    <<cams_net_1_s_[count1]->input_shape->c << " "
                    <<cams_net_1_s_[count1]->input_shape->h << " "
                    <<cams_net_1_s_[count1]->input_shape->w << " "
                    <<std::endl;
                std::cout<<std::endl<<"P_Net---output_shapes[0] "<<
                    output_shapes[0].n<< " "<<
                    output_shapes[0].c<< " "<<
                    output_shapes[0].h<< " "<<
                    output_shapes[0].w<< " "<<std::endl;
                std::cout<<std::endl<<"P-Net---output_shapes[1] "<<
                    output_shapes[1].n<< " "<<
                    output_shapes[1].c<< " "<<
                    output_shapes[1].h<< " "<<
                    output_shapes[1].w<< " "<<
                    std::endl;

                int count = output_shapes[1].count() / 2;
                const float* rect_begin =  output_datas[0];
                const float* rect_end = rect_begin + output_shapes[0].c * count;
                std::vector<float> regression_box_temp_(rect_begin, rect_end);

                const float* confidence_begin = output_datas[1] + count;
                const float* confidence_end = confidence_begin + count;
                std::vector<float> confidence_temp_(confidence_begin, confidence_end);

                std::vector<std::vector<float>> boxes;
                GenerateBoxsNew(img_resized, count1, boxes, confidence_temp_, regression_box_temp_);
                std::cout << "count : " << count1 << " " << "boxes : " << boxes.size() << std::endl;
                std::vector<std::vector<float>> out_boxes;
                nms(boxes, out_boxes, seq, "Union");

                mtx_.lock();
                if(out_boxes.size() > 0)
                    thread_box_.insert(thread_box_.end(), out_boxes.begin(), out_boxes.end());
                mtx_.unlock();

                sem_post(&sem_out_[count1]);
            }
        }

        void FrameStateDetector::P_Net()
        {
            resize_img();
            std::vector<std::vector<float>> total_out_boxes;
            std::vector<int> seq; // no use here
            cout << "++++++++++++++++++++++++++++" << " " << img_resized_.size() << endl;

            thread_box_.clear();
            timePair.clear();
            //auto time_start = std::chrono::steady_clock::now();
            for(int i = 0; i < mparallel_; i++) sem_post(&sem_in_[i]);
            for(int i = 0; i < mparallel_; i++) sem_wait(&sem_out_[i]);
            //auto time_end = std::chrono::steady_clock::now();
            sort(timePair.begin(), timePair.end());
            for(int i = 0; i < mparallel_; i++){
                if(i == mparallel_ - 1) 
                    net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(timePair[i].second - timePair[i].first).count();
                else if(timePair[i].second < timePair[i + 1].first) 
                    net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(timePair[i].second - timePair[i].first).count();
                else timePair[i + 1].first = timePair[i].first;
            }
            for(int i = 0; i < thread_box_.size(); i++) total_out_boxes.push_back(thread_box_[i]);

            nms(total_out_boxes, boxes_, seq, "Union");
            reGetBoxes();
            rerec();
            pad();
            std::cout << "P-Net#############################################END" << params_.size() << std::endl;
        }

        void FrameStateDetector::R_Net()
        {
            detect_net(1);
            std::cout << "R-Net#############################################END" << params_.size() << std::endl;

        }

        void FrameStateDetector::O_Net()
        {
            detect_net(2);
        }

        void FrameStateDetector::cropImages(std::vector<cv::Mat>& cur_imgs, int i)
        {
            for (int j = 0; j < params_.size(); j++) {
                std::vector<float> param = params_[j];
                /*
                 */
                float dy = param[0];
                float edy = param[1];
                float dx = param[2];
                float edx = param[3];
                float y = param[4];
                float ey = param[5];
                float x = param[6];
                float ex = param[7];
                float tmpw = param[8];
                float tmph = param[9];
                //std::cout << "rect " << int(y) << " " << x << " " << int(ey) - int(y) << " " << int(ex) - int(x) << std::endl;
                cv::Rect rect(x, y, ex - x + 1, ey - y + 1);
                cv::Mat src = img_(rect);

                cv::Mat img;
                int top = dy;
                int left = dx;
                if (i == 1)
                    img.create(tmph + 1, tmpw + 1, src.type());
                else if (i == 2)
                    img.create(tmph, tmpw, src.type());

                img.setTo(cv::Scalar::all(0));
                src.copyTo(img(cv::Rect(left, top, src.cols, src.rows)));

                if (img.size() == cv::Size(0, 0))
                    continue;
                if (img.rows == 0 || img.cols == 0)
                    continue;
                if (img.size() != input_geometry_[i])
                    cv::resize(img, img, input_geometry_[i]);
                img.convertTo(img, CV_32FC3, 0.0078125, -127.5*0.0078125);
                cur_imgs.push_back(img);
            }
        }

        void FrameStateDetector::detect_net(int i)
        {
            float thresh = config_.THRESHOLD[i];  //threshold_[i];
            std::vector<cv::Rect> bounding_box;
            std::vector<float> confidence;
            std::vector<cv::Mat> cur_imgs;
            std::vector<std::vector<cv::Point>> alignment;
            //std::cout << "Net " << i << " params_ size=" << params_.size() << std::endl;
            if (params_.size() == 0)
                return;

            cropImages(cur_imgs, i);
            //std::cout << "Net" << i << "cur_imgs size=" << cur_imgs.size() << std::endl;
            if (cur_imgs.size() == 0)
                return;
            Predict(cur_imgs, i);
            std::vector<int> selected;
            getBoxesByScore(thresh, i);
            if (i == 2)
            {
                getPoints();
            }
            std::vector<int> pick;
            if (i == 1)
            {
                //std::cout << "Net" << i << "boxes_ befor nms" << boxes_.size() << std::endl;
                nms(boxes_, boxes_, pick, "Union");
                //std::cout << "Net" << i << "boxes_ after nms" << boxes_.size() << std::endl;
                bbreg(i);
                rerec();
                pad();
            }
            else if (i == 2)
            {
                bbreg(i);
                //std::cout << "Net" << i << "boxes_ befor nms" << boxes_.size() << std::endl;
                nms(boxes_, boxes_, pick, "Min");
                //std::cout << "Net" << i << "boxes_ after nms" << boxes_.size() << std::endl;
                // 去掉已经nms掉的points
                std::vector<std::vector<float>> alignment_t;
                std::vector<float> confidence_t;
                for (size_t j = 0; j < pick.size(); j++)
                {
                    alignment_t.push_back(alignment_[pick[j]]);
                    confidence_t.push_back(confidence_[pick[j]]);
                }
                alignment_ = alignment_t;
                confidence_ = confidence_t;
            }
        }

        void FrameStateDetector::getPoints()
        {
            alignment_.clear();
            for (size_t i = 0; i < boxes_.size(); i++)
            {
                std::vector<float> box = boxes_[i];
                float w = box[3] - box[1] + 1;
                float h = box[2] - box[0] + 1;

                std::vector<float> points;
                for (size_t j = 0; j < 5; j++)
                {
                    float x = w * box[j + 9] + box[0] - 1;
                    float y = h * box[j + 9 + 5] + box[1] - 1;
                    points.push_back(x);
                    points.push_back(y);
                }
                alignment_.push_back(points);
            }
        }

        void FrameStateDetector::getBoxesByScore(float thresh, int count)
        {
            std::vector<std::vector<float>> tmp_boxes;
            tmp_boxes.reserve(confidence_temp_.size() / 2);
            //std::cout << "confidence_temp_.size() :" << confidence_temp_.size() << std::endl;

            for (int j = 0; j < confidence_temp_.size() / 2; j++)
            {
                float score = confidence_temp_[2 * j + 1];
                //std::cout << "getBoxesByScore score=" << score << " thresh:" << thresh << std::endl;
                if (score < thresh) {
                    continue;
                }
                //std::cout << "getBoxesByScore score=" << score << " thresh:" << thresh << std::endl;
                std::vector<float> box(19);
                for (size_t k = 0; k < 4; k++)
                {
                    box[k] = boxes_[j][k];
                }
                box[4] = score;
                for (size_t i = 0; i < 4; i++)
                {
                    box[5 + i] = regression_box_temp_[4 * j + i];
                }
                if (count == 2)
                {
                    for (size_t i = 0; i < 10; i++)
                    {
                        box[9 + i] = alignment_temp_[10 * j + i];
                    }
                }

                tmp_boxes.push_back(box);
            }
            boxes_ = tmp_boxes;
        }

        void FrameStateDetector::bbreg(int count)
        {
            for (size_t i = 0; i < boxes_.size(); i++)
            {
                std::vector<float> box = boxes_[i];
                float w = box[2] - box[0] + 1;
                float h = box[3] - box[1] + 1;

                float bb0 = box[0] + box[5] * w;
                float bb1 = box[1] + box[6] * h;
                float bb2 = box[2] + box[7] * w;
                float bb3 = box[3] + box[8] * h;

                boxes_[i][0] = bb0;
                boxes_[i][1] = bb1;
                boxes_[i][2] = bb2;
                boxes_[i][3] = bb3;

                if (count == 2) {
                    confidence_.push_back(boxes_[i][4]);
                }
            }
        }

        // 倒序排列
        void  FrameStateDetector::sort_vector(std::vector<std::vector<float>>& boxes, std::vector<int>& seq)
        {
            if (boxes.size() == 0) return;

            // 记录之前的顺序
            seq.resize(boxes.size());
            for (size_t i = 0; i < boxes.size(); i++)
            {
                seq[i] = i;
            }

            for (size_t i = 0; i < boxes.size(); i++)
            {
                for (size_t j = i + 1; j < boxes.size(); j++) {
                    if (boxes[i][4] < boxes[j][4]) {
                        std::vector<float> t = boxes[j];
                        boxes[j] = boxes[i];
                        boxes[i] = t;

                        int tt = seq[j];
                        seq[j] = seq[i];
                        seq[i] = tt;
                    }
                }
            }
        }

        void FrameStateDetector::nms(std::vector<std::vector<float>> boxes, std::vector<std::vector<float>>& out_boxes, std::vector<int>& left, std::string type)
        {
            out_boxes.clear();
            double threshold = config_.THRESHOLD_NMS; //threshold_NMS_;
            std::vector<int> seq;
            sort_vector(boxes, seq);
            while (boxes.size() > 0)
            {
                int j = 0; // 最大值
                for (size_t i = 1; ; i++)
                {
                    if (i >= boxes.size()) break;

                    double a = IoU(boxes[i], boxes[j], type);
                    if (a > threshold)
                    {
                        boxes.erase(boxes.begin() + i);
                        seq.erase(seq.begin() + i);
                        i--;
                    }
                }
                out_boxes.push_back(boxes[j]);
                left.push_back(seq[j]);
                boxes.erase(boxes.begin() + j); //去掉最高的
                seq.erase(seq.begin() + j);
            }
        }

        void FrameStateDetector::reGetBoxes()
        {
            for (size_t i = 0; i < boxes_.size(); i++)
            {
                std::vector<float> box = boxes_[i];
                float regh = box[3] - box[1];
                float regw = box[2] - box[0];
                float t1 = box[0] + box[5] * regw;
                float t2 = box[1] + box[6] * regh;
                float t3 = box[2] + box[7] * regw;
                float t4 = box[3] + box[8] * regh;
                float t5 = box[4];
                std::vector<float> new_box;
                new_box.push_back(t1);
                new_box.push_back(t2);
                new_box.push_back(t3);
                new_box.push_back(t4);
                new_box.push_back(t5);
                boxes_[i] = new_box;
            }
        }

        void FrameStateDetector::rerec()
        {
            for (size_t i = 0; i < boxes_.size(); i++)
            {
                std::vector<float> box = boxes_[i];
                float w = box[2] - box[0];
                float h = box[3] - box[1];
                float l = w > h ? w : h;
                box[0] = box[0] + w*0.5 - l*0.5;
                box[1] = box[1] + h*0.5 - l*0.5;
                box[2] = box[0] + l;
                box[3] = box[1] + l;
                // C++ fix
                box[0] = box[0] > 0 ? floor(box[0]) : ceil(box[0]);
                box[1] = box[1] > 0 ? floor(box[1]) : ceil(box[1]);
                box[2] = box[2] > 0 ? floor(box[2]) : ceil(box[2]);
                box[3] = box[3] > 0 ? floor(box[3]) : ceil(box[3]);

                boxes_[i] = box;
            }
        }

        float FrameStateDetector::maximum(float f1, float f2)
        {
            return f1 > f2 ? f1 : f2;
        }

        //int FrameStateDetector::maximum(int f1, int f2)
        //{
        //	return f1 > f2 ? f1 : f2;
        //}

        int FrameStateDetector::minimum(int f1, int f2)
        {
            return f1 < f2 ? f1 : f2;
        }

        void FrameStateDetector::pad()
        {
            float w = img_.cols;
            float h = img_.rows;
            params_.clear();
            params_.resize(boxes_.size());
            //std::cout << "boxes_: " << boxes_.size() << std::endl;
            for (size_t i = 0; i < boxes_.size(); i++)
            {
                std::vector<float> box = boxes_[i];
                float tmph = box[3] - box[1] + 1;
                float tmpw = box[2] - box[0] + 1;
                float dx = 1.;
                float dy = 1.;
                float edx = tmpw;
                float edy = tmph;

                float x = box[0];
                float y = box[1];
                float ex = box[2];
                float ey = box[3];

                if (ex > w)
                {
                    edx = -ex + w - 1 + tmpw;
                    ex = w - 1;
                }


                if (ey > h) {
                    edy = -ey + h - 1 + tmph;
                    ey = h - 1;
                }
                if (x < 1) {

                    dx = 2 - x;
                    x = 1.;
                }
                if (y < 1) {
                    dy = 2 - y;
                    y = 1.;
                }

                dy = maximum(0, dy - 1);
                dx = maximum(0, dx - 1);
                y = maximum(0, y - 1);
                x = maximum(0, x - 1);
                edy = maximum(0, edy - 1);
                edx = maximum(0, edx - 1);
                ey = maximum(0, ey - 1);
                ex = maximum(0, ex - 1);
                std::vector<float> result;
                result.resize(10);
                result[0] = dy;
                result[1] = edy;
                result[2] = dx;
                result[3] = edx;
                result[4] = y;
                result[5] = ey;
                result[6] = x;
                result[7] = ex;
                result[8] = edx + 1;// tmpw; python中此处是引用传递，讲tmpw指向了edx，而c++是值传递
                result[9] = edy + 1;// tmph;
                params_[i] = result;
            }
        }

        /*
         * Predict function input is a image without crop
         * the reshape of input layer is image's height and width
         */
        void FrameStateDetector::Predict(const cv::Mat& img, int i, int count1, vector<float>& confidence_temp_, vector<float>& regression_box_temp_)
        {
            if (img.cols * img.rows == 0)
            {
                BOOST_LOG_TRIVIAL(error) << "Input image is empty.";
                return;
            }

            //CambriconUtil* camPnet = cams_net_1_s_[count1];

            CambriconUtil* camPnet = new CambriconUtil();
            camPnet->init_net(binary_inputs_[2 * count1], binary_inputs_[2 * count1 + 1]);

            cv::Mat cls_blob_img = blobFromImage(img.t(), 1.0f, false);
            std::vector<float*> output_datas;
            std::vector<MShape> output_shapes;
            int time_count = 0;
            auto time_start = std::chrono::steady_clock::now();
            //cams_net_1_s_[count1]->run(cls_blob_img.ptr<float>(0), output_datas, output_shapes);
            camPnet->run(cls_blob_img.ptr<float>(0), output_datas, output_shapes);
            auto time_end = std::chrono::steady_clock::now();
            net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();

            BOOST_LOG_TRIVIAL(debug) << "Forward P-net: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count() << " ms";
            std::cout<<"P_Net---input_shapes[0] "
                <<cams_net_1_s_[count1]->input_shape->n << " "
                <<cams_net_1_s_[count1]->input_shape->c << " "
                <<cams_net_1_s_[count1]->input_shape->h << " "
                <<cams_net_1_s_[count1]->input_shape->w << " "
                <<std::endl;
            std::cout<<std::endl<<"P_Net---output_shapes[0] "<<
                output_shapes[0].n<< " "<<
                output_shapes[0].c<< " "<<
                output_shapes[0].h<< " "<<
                output_shapes[0].w<< " "<<std::endl;
            std::cout<<std::endl<<"P-Net---output_shapes[1] "<<
                output_shapes[1].n<< " "<<
                output_shapes[1].c<< " "<<
                output_shapes[1].h<< " "<<
                output_shapes[1].w<< " "<<
                std::endl;

            int count = output_shapes[1].count() / 2;
            const float* rect_begin =  output_datas[0];
            const float* rect_end = rect_begin + output_shapes[0].c * count;
            regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

            const float* confidence_begin = output_datas[1] + count;
            const float* confidence_end = confidence_begin + count;
            confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
        }

        /*
         * Predict(const std::vector<cv::Mat> imgs, int i) function
         * used to input is a group of image with crop from original image
         * the reshape of input layer of net is the number, channels, height and width of images.
         */

        void Get_vector(float* my_begin, int len, vector<float>& s)
        {
            float* my_end = my_begin + len;
            while(my_begin != my_end){
                s.push_back(*my_begin);
                my_begin++;
            }  
        }

        void FrameStateDetector::doThread(int i, int now, const vector<cv::Mat> &new_imgs, int start, int len, int CONF_INDEX, int RESS_INDEX, int ALIG_INDEX, int& time_cost)
        {
            const int iteration_size = 8;

            for(int t = start; t <= start + len - iteration_size; t += iteration_size){
                vector<cv::Mat> my_imgs;
                for(int j = t; j < t + iteration_size; j++) my_imgs.push_back(new_imgs[j]);
                cv::Mat cls_blob_img = blobFromImages(my_imgs, 1.0f, false);
                std::vector<float*> output_datas;
                std::vector<MShape> output_shapes;

                int time_count = 0;
                auto time_start = std::chrono::steady_clock::now();
                _myCams_[i]->my_run(cls_blob_img.ptr<float>(0), output_datas, output_shapes, my_imgs.size());
                auto time_end = std::chrono::steady_clock::now();
                net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();
                time_cost += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();

                int count = output_shapes[CONF_INDEX].count() / 2;
                std::cout<<std::endl<< i <<" CONF_INDEX  count=  "<< output_shapes[CONF_INDEX].count() << " " << t << " " << new_imgs.size() << endl;

                Get_vector(output_datas[CONF_INDEX], count * 2, confidence_temp_);

                Get_vector(output_datas[RESS_INDEX], output_shapes[RESS_INDEX].c * count, regression_box_temp_);

                if (i == 2)  Get_vector(output_datas[ALIG_INDEX], output_shapes[ALIG_INDEX].c * count, alignment_temp_);
            }

            for(int t = new_imgs.size() / iteration_size * iteration_size; t < new_imgs.size(); t++){
                cv::Mat cls_blob_img = blobFromImage(new_imgs[t], 1.0f, false);
                std::vector<float*> output_datas;
                std::vector<MShape> output_shapes;

                int time_count = 0;
                auto time_start = std::chrono::steady_clock::now();
                lcams_[i]->run(cls_blob_img.ptr<float>(0), output_datas, output_shapes);
                auto time_end = std::chrono::steady_clock::now();
                net_time_count += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();
                time_cost += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start ).count();

                int count = output_shapes[CONF_INDEX].count() / 2;
                std::cout<<std::endl<< i <<" CONF_INDEX  count=  "<< output_shapes[CONF_INDEX].count() << " " << t << " " << new_imgs.size() << endl;

                Get_vector(output_datas[CONF_INDEX], count * 2, confidence_temp_);

                Get_vector(output_datas[RESS_INDEX], output_shapes[RESS_INDEX].c * count, regression_box_temp_);

                if (i == 2)  Get_vector(output_datas[ALIG_INDEX], output_shapes[ALIG_INDEX].c * count, alignment_temp_);
            }
        }

        void FrameStateDetector::Predict(const std::vector<cv::Mat>& imgs, int i)
        {
            int CONF_INDEX = (i == 1 ? 1 : 2);
            int RESS_INDEX = (i == 1 ? 0 : 0);
            int ALIG_INDEX = 1;
            if (imgs.empty())
            {
                BOOST_LOG_TRIVIAL(error) << "Input images is empty.";
                return;
            }
            clock_t start = clock();
            double cost_time = (double(clock() - start)) / CLOCKS_PER_SEC * 1000;
            confidence_temp_.clear(), regression_box_temp_.clear(), alignment_temp_.clear();
            vector<cv::Mat>new_imgs;
            for(int i = 0; i < imgs.size(); i++) new_imgs.push_back(imgs[i].t());
            std::cout << "images :    " << i << " " << new_imgs.size() << std::endl;
            //auto time_start = std::chrono::steady_clock::now();
            int time_cost = 0;
            doThread(i, 0, new_imgs, 0, new_imgs.size(), CONF_INDEX, RESS_INDEX, ALIG_INDEX, time_cost);
            //auto time_end = std::chrono::steady_clock::now();

            if (i == 1)
                BOOST_LOG_TRIVIAL(debug) << "Forward R-net: " << time_cost << " ms";
            else if (i == 2)
                BOOST_LOG_TRIVIAL(debug) << "Forward O-net: " << time_cost << " ms";
        }

        /*void FrameStateDetector::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
          {
          auto input_layer = nets_[i]->blob("data");

          int width = input_layer->width();
          int height = input_layer->height();
          float* input_data = input_layer->mutable_cpu_data();
          for (int j = 0; j < input_layer->channels(); ++j)
          {
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels->push_back(channel);
          input_data += width * height;
          }

          cv::split(img.t(), *input_channels);

          }


         * WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) function
         * used to write the separate BGR planes directly to the input layer of the network

         void FrameStateDetector::WrapInputLayer(const std::vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i)
         {
        //Blob<float> *input_layer = nets_[i]->input_blobs()[0];
        auto input_layer = nets_[i]->blob("data");

        int width = input_layer->width();
        int height = input_layer->height();
        int num = input_layer->num();
        float *input_data = input_layer->mutable_cpu_data();

        for (int j = 0; j < num; j++) {
        for (int k = 0; k < input_layer->channels(); ++k) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
        }
        cv::Mat img = imgs[j].t();
        cv::split(img, *input_channels);
        input_channels->clear();
        }
        }*/

        double FrameStateDetector::IoU(const std::vector<float>& rect1, const std::vector<float>& rect2, std::string type)
        {
            int xx1 = maximum(rect1[0], rect2[0]);
            int yy1 = maximum(rect1[1], rect2[1]);
            int xx2 = minimum(rect1[2], rect2[2]);
            int yy2 = minimum(rect1[3], rect2[3]);
            int w = maximum(0, xx2 - xx1 + 1);
            int h = maximum(0, yy2 - yy1 + 1);
            int inter = w * h;
            int area1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1);
            int area2 = (rect2[2] - rect2[0] + 1) * (rect2[3] - rect2[1] + 1);
            if (type == "Min") {
                int l = area1 > area2 ? area1 : area2;
                int s = area1 < area2 ? area1 : area2;
                return 1.0 * inter / s;
            }
            else {
                double o = (double)inter / ((double)area1 + (double)area2 - (double)inter);
                return o;
            }
        }

        void FrameStateDetector::resize_img()
        {
            cv::Mat img = img_;
            long height = (long)img.rows;
            long width = (long)img.cols;

            int minSize = config_.MINSIZE;
            double factor = config_.FACTOR;

            double scale = (double)12. / minSize;
            int minWH = std::min(height, width) * scale;

            std::vector<cv::Mat> img_resized;

            int count = 0;
            while (minWH >= 12)
            {
                double p = pow(factor, count);
                double fh = height * scale * p;
                double fw = width * scale * p;
                int resized_h = (int)ceil(fh);
                int resized_w = (int)ceil(fw);
                //printf("p is %0.17f\n", scale * p);
                //printf("h is %0.17f\n", fh);
                //std::cout << "######################################" << resized_h << " " << resized_w << std::endl;
                /*if (count == 0) {
                  resized_h = 540; resized_w = 960;
                  }else if (count == 1) {
                  resized_h = 37; resized_w =49;
                  }
                  else if (count == 2) {
                  resized_h = 27; resized_w = 36;
                  }
                  else if (count == 3) {
                  resized_h = 21; resized_w = 27;
                  }
                  else if (count == 4) {
                  resized_h = 16; resized_w = 21;
                  }*/
                cv::Mat resized;
                cv::resize(img, resized, cv::Size(resized_w, resized_h));
                resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);
                img_resized.push_back(resized);

                scales_.push_back(scale * pow(factor, count));
                minWH *= factor;
                count++;
            }
            img_resized_ = img_resized;
        }

        void FrameStateDetector::GenerateBoxsNew(const cv::Mat& img, int count, std::vector<std::vector<float>>& boxes, vector<float>confidence_temp_, vector<float>regression_box_temp_)
        {
            boxes.clear();
            int image_h = img.rows;
            int image_w = img.cols;
            //std::cout << "image_h=" << image_h << "image_w" << image_w <<std::endl;
            double thresh = config_.THRESHOLD[0];  //threshold_[0];
            int stride = 2;
            int cellSize = input_geometry_[0].width;
            double scale = scales_[count];
            std::cout << "scale=" << scale << "cellSize=" << cellSize <<std::endl;

            int feature_map_h = std::ceil((double)(image_h - cellSize)*1.0 / stride) + 1;
            int feature_map_w = std::ceil((double)(image_w - cellSize)*1.0 / stride) + 1;
            //std::cout << "feature_map_h=" << feature_map_h << "feature_map_w" << feature_map_w <<std::endl;
            int layer_count = feature_map_h * feature_map_w;
            for (size_t i = 0; i < feature_map_h; i++)
            {
                double socre = 0;
                for (size_t j = 0; j < feature_map_w; j++) {
                    socre = confidence_temp_[j * feature_map_h + i];
                    if (socre < thresh)
                        continue;

                    double ddx1 = (double)(stride * j + 1) / (double)scale;
                    double ddy1 = (double)(stride * i + 1) / (double)scale;
                    double ddx2 = (double)(stride * j + cellSize) / (double)scale;
                    double ddy2 = (double)(stride * i + cellSize) / (double)scale;

                    float x1 = std::floor(ddx1);
                    float y1 = std::floor(ddy1);
                    float x2 = std::floor(ddx2);
                    float y2 = std::floor(ddy2);

                    float dx1 = regression_box_temp_[j * feature_map_h + i + layer_count * 0];
                    float dy1 = regression_box_temp_[j * feature_map_h + i + layer_count * 1];
                    float dx2 = regression_box_temp_[j * feature_map_h + i + layer_count * 2];
                    float dy2 = regression_box_temp_[j * feature_map_h + i + layer_count * 3];

                    std::vector<float> one_box;
                    one_box.reserve(9);
                    one_box.push_back(x1);
                    one_box.push_back(y1);
                    one_box.push_back(x2);
                    one_box.push_back(y2);
                    one_box.push_back(socre);
                    one_box.push_back(dx1);
                    one_box.push_back(dy1);
                    one_box.push_back(dx2);
                    one_box.push_back(dy2);
                    boxes.push_back(one_box);
                }
            }
        }
    }
}
