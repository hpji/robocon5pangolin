#include <pangolin/pangolin.h>
#include "header.h"
#include "camera.h"
#include "arrow.h"
#include "zhencha.h"
#include "dianxuan.h"
#include "jiaozheng.h"

class PGL
{
    private:

    public:
        PGL();
        ~PGL();

        void get_loc(Mat pixel_l, Mat pixel_r, Mat &a, Mat &b);
        void arrowshibie(Camera cam0,Camera cam1);
        void tennisshibie(Camera cam0,Camera cam1);
        void UI_all(Camera cam0,Camera cam1);

        ArrowDetect detect_l, detect_r;
        zhencha zctennis;
        cv::Mat img1,img2;
        cv::Point3f point,point_jz;
        std::vector<cv::Point3f> points;
        std::vector<pangolin::Var<float>> data_set;
        
};
