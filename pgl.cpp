#include "pgl.h"

bool arrow_shibie = false;
bool tennis_shibie = false;
bool if_start = false;
bool if_end = false;
bool if_clear = false;
bool if_out = false;
bool if_up = false;
bool if_down = false;
bool if_xuandian=false;

int fnamenum = 0;
ofstream outfile;
cv::Scalar pcolor(0, 255, 0);
cv::Point2f mappoint;

//相机参数
//7.10
cv::Mat cameraMatrixL = (Mat_<double>(3, 3) << 1191.459, 0, 686.2256,
                         0, 1188.452, 571.7904,
                         0, 0, 1);

cv::Mat cameraMatrixR = (Mat_<double>(3, 3) << 1191.289, 0, 721.6954,
                         0, 1188.601, 577.5508,
                         0, 0, 1);

cv::Mat T = (Mat_<double>(3, 1) << -516.2102, 1.0273, -8.9323);

cv::Mat R = (Mat_<double>(3, 3) << 0.999751, 0.001238, -0.022262,
             -0.000809, 0.999813, 0.019305,
             0.022283, -0.019282, 0.999566);

cv::Mat distortionCoeffL = (Mat_<double>(1, 5) << -0.128205070001341, 0.294578913559712,
                            -0.0016518365848735, -0.000105206218972814, -0.711284075326086);

cv::Mat distortionCoeffR = (Mat_<double>(1, 5) << -0.112070486580824, -0.0352243116180731,
                            -0.00069996633266156, 0.000726626214510611, 0.601223421015994);

// cv::Mat cameraMatrixL = (Mat_<double>(3, 3) << 1222.7305, 0, 700.4029,
//                          0, 1219.6568, 570.6165,
//                          0, 0, 1);
// cv::Mat cameraMatrixR = (Mat_<double>(3, 3) << 1222.0738, 0, 725.4911,
//                          0, 1217.9931, 577.0063,
//                          0, 0, 1);

// cv::Mat T = (Mat_<double>(3, 1) << -513.3349, 1.2873, -5.9023);

// cv::Mat R = (Mat_<double>(3, 3) << 0.999985, -0.000658, -0.005380,
//              0.000725, 0.999923, 0.012376,
//              0.005372, -0.012379, 0.999908);

//识别 相关变量
cv::Mat org_img1,org_img2;
cv::Mat pr, Extrinsic;
cv::Mat pix_l, pix_r;
cv::Mat l_small, r_small;
cv::Mat l_all, r_all;
cv::Mat l_all_jz,r_all_jz;
cv::Point2f ans_l, ans_r;
std::vector<cv::Point> group_l, group_r;
std::vector<cv::Point> grp_l, grp_r;
std::vector<cv::Point> record_l, record_r;
std::vector<cv::Point3f> linshi_p;

cv::Mat changdibmp=cv::imread("/home/gohu3/PGLUI3.0/map.bmp");
cv::Mat org_changdibmp,org_heightbmp;
cv::VideoCapture cp0("/home/gohu3/桌面/camnew/build/3/L1.avi");
cv::VideoCapture cp1("/home/gohu3/桌面/camnew/build/3/R1.avi");

PGL::PGL()
{
}

PGL::~PGL()
{
}

int max(int a, int b)
{
    if (a < b)
    {
        return b;
    }
    else
    {
        return a;
    }
}

int min(int a, int b)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

void PGL::get_loc(Mat pixel_l, Mat pixel_r, Mat &a, Mat &b)
{
    double kxz, kyz;
    kxz = (pixel_l.at<double>(0, 0) - cameraMatrixL.at<double>(0, 2)) / cameraMatrixL.at<double>(0, 0);
    kyz = (pixel_l.at<double>(1, 0) - cameraMatrixL.at<double>(1, 2)) / cameraMatrixL.at<double>(1, 1);

    double zl;

    double up = pr.at<double>(1, 3) / pixel_r.at<double>(1, 0) - pr.at<double>(0, 3) / pixel_r.at<double>(0, 0);
    double down = (pr.at<double>(0, 0) * kxz + pr.at<double>(0, 1) * kyz + pr.at<double>(0, 2)) / pixel_r.at<double>(0, 0) - (pr.at<double>(1, 0) * kxz + pr.at<double>(1, 1) * kyz + pr.at<double>(1, 2)) / pixel_r.at<double>(1, 0);
    zl = up / down;
    a = (Mat_<double>(3, 1) << zl * kxz, zl * kyz, zl);
    b = R * a + T;
}

void arrow_method()
{
    std::cout << "begin arrow" << std::endl;
    arrow_shibie = true;
    tennis_shibie=false;
}

void tennis_method()
{
    std::cout << "begin tennis" << std::endl;
    tennis_shibie = true;
    arrow_shibie=false;
}

void start_method()
{
    std::cout << "start to record" << std::endl;
    pcolor = cv::Scalar(0, 255, 255);
    if_start = true;
    if_end = false;
}

void end_method()
{
    std::cout << "finish recording" << std::endl;
    pcolor = cv::Scalar(0, 255, 0);
    if_end = true;
    if_start = false;
}

void clear_method()
{
    std::cout << "clear screen" << std::endl;
    if_clear = true;
}

void output_method()
{
    if_out = true;
    std::string filename = std::to_string(fnamenum);
    filename = "data" + filename + ".txt";
    outfile.open(filename, ios::trunc);
}

void gain_u()
{
    if_up = true;
}

void gain_d()
{
    if_down = true;
}

//识别程序
void PGL::arrowshibie(Camera cam0, Camera cam1)
{
    //输入图像
    //输入视频图像
    // cp0>>this->img1;
    // cp1>>this->img2;
    //输入摄像头图像
    cam0.getImage(this->img1);
    cam1.getImage(this->img2);

    cv::resize(this->img1, l_small, cv::Size(), 0.5, 0.5);
    cv::resize(this->img2, r_small, cv::Size(), 0.5, 0.5);

    this->detect_l.find_arrowhead(l_small);
    this->detect_r.find_arrowhead(r_small);

    ans_l = detect_l.head;
    ans_r = detect_r.head;
    ans_l = cv::Point(ans_l.x * 2, ans_l.y * 2);
    ans_r = cv::Point(ans_r.x * 2, ans_r.y * 2);

    group_l.push_back(ans_l);
    group_r.push_back(ans_r);

    for (int i = group_l.size() - 1; i >= max(0, group_l.size() - 20); i--)
    {
        cv::circle(l_small, cv::Point2f(group_l[i].x / 2, group_l[i].y / 2), 3, pcolor, -1);
        cv::circle(r_small, cv::Point2f(group_r[i].x / 2, group_r[i].y / 2), 3, pcolor, -1);
    }

    if(if_start)
    {
        grp_l.push_back(ans_l);
        grp_r.push_back(ans_r);
        for (int i = 0; i < grp_l.size(); i++)
        {
            cv::circle(l_small, cv::Point2f(grp_l[i].x / 2, grp_l[i].y / 2), 3, pcolor, -1);
            cv::circle(r_small, cv::Point2f(grp_r[i].x / 2, grp_r[i].y / 2), 3, pcolor, -1);
        }
    }
    else if (if_end)
    {
        for (int i = group_l.size() - 1; i >= max(0, group_l.size() - 20); i--)
        {
            cv::circle(l_small, cv::Point2f(group_l[i].x / 2, group_l[i].y / 2), 3, pcolor, -1);
            cv::circle(r_small, cv::Point2f(group_r[i].x / 2, group_r[i].y / 2), 3, pcolor, -1);
        }
        for (int i = 0; i < grp_l.size(); i++)
        {
            cv::circle(l_small, cv::Point2f(grp_l[i].x / 2, grp_l[i].y / 2), 3, cv::Scalar(0,255,255), -1);
            cv::circle(r_small, cv::Point2f(grp_r[i].x / 2, grp_r[i].y / 2), 3, cv::Scalar(0,255,255), -1);
        }
    }
    else
    {
        for (int i = group_l.size() - 1; i >= max(0, group_l.size() - 20); i--)
        {
            cv::circle(l_small, cv::Point2f(group_l[i].x / 2, group_l[i].y / 2), 3, pcolor, -1);
            cv::circle(r_small, cv::Point2f(group_r[i].x / 2, group_r[i].y / 2), 3, pcolor, -1);
        }
        grp_l.clear();
        grp_r.clear();
    }   

    if (this->detect_l.head != cv::Point(0, 0) && this->detect_r.head != cv::Point(0, 0))
    {
        pix_l = (cv::Mat_<double>(3, 1) << ans_l.x, ans_l.y, 1);
        pix_r = (cv::Mat_<double>(3, 1) << ans_r.x, ans_r.y, 1);
        this->get_loc(pix_l, pix_r, l_all, r_all);

        //以左相机图像坐标系中的x、y、z坐标构建三维坐标
        cv::Mat middle=l_all - cam0.T;
        cv::Mat R_ni;
        invert(cam0.R,R_ni);
        Mat point_3d=R_ni*middle;
        if(cam0.l_r)
        {
            swap(point_3d.at<double>(0, 0),point_3d.at<double>(0, 2));
        }
       this->point = cv::Point3f(point_3d.at<double>(0, 0), point_3d.at<double>(0, 1), point_3d.at<double>(0, 02));
       cv::Mat start = (cv::Mat_<double>(3, 1) << 0, 0, 0);

        this->data_set[0] = this->point.x;
        this->data_set[1] = this->point.y;
        this->data_set[2] = this->point.z;

        linshi_p.push_back(this->point);

        if (if_start == true)
        {
            this->points.push_back(this->point);
            record_l.push_back(ans_l);
            record_r.push_back(ans_r);
        }
    }
}

void PGL::tennisshibie(Camera cam0,Camera cam1)
{
    //输入图像
    //输入视频图像
    // cp0>>this->img1;
    // cp1>>this->img2;
    //输入摄像头图像
    cam0.getImage(this->img1);
    cam1.getImage(this->img2);

    this->zctennis.gshibie(this->img1,this->zctennis.mask1);
    this->zctennis.gshibie(this->img2,this->zctennis.mask2);

    this->zctennis.zhencha_2(this->img1,this->img2);
    this->zctennis.mixture();

    ans_l = this->zctennis.gcenter1;
    ans_r = this->zctennis.gcenter2;

    group_l.push_back(ans_l);
    group_r.push_back(ans_r);

    if(if_start)
    {
        grp_l.push_back(ans_l);
        grp_r.push_back(ans_r);
        for (int i = 0; i < grp_l.size(); i++)
        {
            cv::circle(this->img1, cv::Point2f(grp_l[i].x, grp_l[i].y), 11, pcolor, -1);
            cv::circle(this->img2, cv::Point2f(grp_r[i].x, grp_r[i].y), 11, pcolor, -1);
        }
    }
    else if (if_end)
    {
        for (int i = (group_l.size() - 1); i>(group_l.size() - 2); i--)
        {
            cv::circle(this->img1, cv::Point2f(group_l[i].x, group_l[i].y), 11, pcolor, -1);
            cv::circle(this->img2, cv::Point2f(group_r[i].x, group_r[i].y), 11, pcolor, -1);
        }
        for (int i = 0; i < grp_l.size(); i++)
        {
            cv::circle(this->img1, cv::Point2f(grp_l[i].x, grp_l[i].y), 11, cv::Scalar(0,255,255), -1);
            cv::circle(this->img2, cv::Point2f(grp_r[i].x, grp_r[i].y), 11, cv::Scalar(0,255,255), -1);
        }
    }
    else
    {
        for (int i = (group_l.size() - 1); i>(group_l.size() - 2); i--)
        {
            cv::circle(this->img1, cv::Point2f(group_l[i].x, group_l[i].y), 13, pcolor, -1);
            cv::circle(this->img2, cv::Point2f(group_r[i].x, group_r[i].y), 13, pcolor, -1);
        }
        grp_l.clear();
        grp_r.clear();
    }

    if (this->zctennis.gcenter1 != cv::Point2f(0, 0) && this->zctennis.gcenter2 != cv::Point2f(0, 0))
    {
        pix_l = (cv::Mat_<double>(3, 1) << ans_l.x, ans_l.y, 1);
        pix_r = (cv::Mat_<double>(3, 1) << ans_r.x, ans_r.y, 1);

        this->get_loc(pix_l, pix_r, l_all, r_all);

        cv::Point2f ans_l_jz, ans_r_jz;
        myUndistortPoints(ans_l, ans_l_jz, cameraMatrixL, distortionCoeffL);
        myUndistortPoints(ans_r, ans_r_jz, cameraMatrixR, distortionCoeffR);
                
        pix_l = (cv::Mat_<double>(3, 1) << ans_l_jz.x, ans_l_jz.y, 1);
        pix_r = (cv::Mat_<double>(3, 1) << ans_r_jz.x, ans_r_jz.y, 1);

        this->get_loc(pix_l, pix_r, l_all_jz, r_all_jz);

        //以左相机图像坐标系中的x、y、z坐标构建三维坐标
        cv::Mat middle = l_all - cam0.T;
        cv::Mat R_ni;
        cv::invert(cam0.R, R_ni);
        cv::Mat point_3d = R_ni*middle;
        if(cam0.l_r)
        {
            swap(point_3d.at<double>(0, 0),point_3d.at<double>(0, 2));
        }
        if (this->point.z <= 60)
        {
            this->point.z = 30;
        }
       this->point = cv::Point3f(point_3d.at<double>(0, 0), point_3d.at<double>(0, 1), point_3d.at<double>(0, 02));
       cv::Mat start = (cv::Mat_<double>(3, 1) << 0, 0, 0);

//矫正
        cv::Mat middle_jz = l_all_jz - cam0.T;
        cv::Mat point_3d_jz = R_ni * middle_jz;
        if (cam0.l_r)
        {
            swap(point_3d_jz.at<double>(0, 0), point_3d_jz.at<double>(0, 2));
        }
        this->point_jz = cv::Point3f(point_3d_jz.at<double>(0, 0), point_3d_jz.at<double>(0, 1), point_3d_jz.at<double>(0, 2));

        //显示当前点的坐标
        this->data_set[0] = this->point.x;
        this->data_set[1] = this->point.y;
        this->data_set[2] = this->point.z;

        linshi_p.push_back(this->point);

        if (if_start == true && if_end == false)
        {
            this->points.push_back(this->point);
            record_l.push_back(ans_l);
            record_r.push_back(ans_r);
        }
    }   
}

void PGL::UI_all(Camera cam0, Camera cam1)
{
    pangolin::CreateWindowAndBind("rccv_main", 1440, 1080);
    glEnable(GL_DEPTH_TEST);

    //创建观察摄影机
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1440, 1080, 420, 420, 320, 320, 0.1, 1000),
        pangolin::ModelViewLookAt(-2, -2, -2, 0, 0, 0, pangolin::AxisY));

    //创建交互界面
    pangolin::View &d_cam = pangolin::Display("visual_cam")
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(360), 1.0, -1440 / 1080.)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    //定义控制面板
    pangolin::CreatePanel("ui")
        .SetBounds(pangolin::Attach::Pix(720), 1.0, 0.0, pangolin::Attach::Pix(360));

    //定义数据面板
    pangolin::CreatePanel("data")
        .SetBounds(pangolin::Attach::Pix(540), pangolin::Attach::Pix(720), 0., pangolin::Attach::Pix(360), 1440 / 1080.);

    //设置按钮
    pangolin::Var<std::function<void()>> Arrowshibie_btm("ui.arrow_shibie", arrow_method);
    pangolin::Var<std::function<void()>> Tennisshibie_btm("ui.tennis_shibie", tennis_method);
    pangolin::Var<std::function<void()>> Start_btm("ui.start", start_method);
    pangolin::Var<std::function<void()>> End_btm("ui.end", end_method);
    pangolin::Var<std::function<void()>> Clear_btm("ui.clear", clear_method);
    pangolin::Var<std::function<void()>> Output_btm("ui.output", output_method);
    pangolin::Var<std::function<void()>> Up_btm("ui.up", gain_u);
    pangolin::Var<std::function<void()>> Down_btm("ui.down", gain_d);

    //显示数据
    pangolin::Var<float> curr_X("data.X");
    pangolin::Var<float> curr_Y("data.Y");
    pangolin::Var<float> curr_Z("data.Z");
    pangolin::Var<float> curr_gain("data.gain");
    pangolin::Var<float> curr_exposure("data.exposure");
    this->data_set.push_back(curr_X);
    this->data_set.push_back(curr_Y);
    this->data_set.push_back(curr_Z);
    this->data_set.push_back(curr_gain);
    this->data_set.push_back(curr_exposure);

    //设置相机视图
    pangolin::View &cv_img_1 = pangolin::Display("image_1")
                                   .SetBounds(pangolin::Attach::Pix(270), pangolin::Attach::Pix(540), 0., pangolin::Attach::Pix(360), 720 / 540.)
                                   .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    pangolin::View &cv_img_2 = pangolin::Display("image_2")
                                   .SetBounds(0., pangolin::Attach::Pix(360), 0, pangolin::Attach::Pix(360), 720 / 540.)
                                   .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    pangolin::GlTexture imgTexture1(720, 540, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
    pangolin::GlTexture imgTexture2(720, 540, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
    //设置场地图
    pangolin::View &changdi_map = pangolin::Display("changdimap")
                                    .SetBounds(0., pangolin::Attach::Pix(600), pangolin::Attach::Pix(600), 1., 1200 / 1200.)
                                    .SetLock(pangolin::LockRight,pangolin::LockBottom);
    pangolin::GlTexture changdi(1200, 1200, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

    cv::hconcat(R, T, Extrinsic);
    pr = cameraMatrixR * Extrinsic;

    //载入初始图像
    //载入视频
    // cp0>>this->img1;
    // cp1>>this->img2;
    //载入摄像头
    cam0.getImage(this->img1);
    cam1.getImage(this->img2);
    this->detect_l.first_frame(this->img1);
    this->detect_r.first_frame(this->img2);
    this->zctennis.pframe_l=this->img1;
    this->zctennis.pframe_r=this->img2;

    time_t t0 = time(0);
    time_t t_now;
    int fps = 0;
    auto exat_last = chrono::system_clock::now();
    auto op_stime = chrono::system_clock::now();
    auto op_time = chrono::duration_cast<chrono::microseconds>(op_stime - exat_last);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        this->data_set[3] = cam0.gain;
        this->data_set[4] = cam0.exposureTime;
        
        if (arrow_shibie == false && tennis_shibie == false)
        {
            //载入视频
            // cp0>>this->img1;
            // cp1>>this->img2;
            //载入摄像头
            cam0.getImage(this->img1);
            cam1.getImage(this->img2);

            cv::resize(this->img1, this->img1, cv::Size(), 0.5, 0.5);
            cv::resize(this->img2, this->img2, cv::Size(), 0.5, 0.5);

            imgTexture1.Upload(this->img1.data, GL_BGR, GL_UNSIGNED_BYTE);
            imgTexture2.Upload(this->img2.data, GL_BGR, GL_UNSIGNED_BYTE);
        }
        else if(arrow_shibie)
        {
            this->arrowshibie(cam0, cam1);

            imgTexture1.Upload(l_small.data, GL_BGR, GL_UNSIGNED_BYTE);
            imgTexture2.Upload(r_small.data, GL_BGR, GL_UNSIGNED_BYTE);
        }
        else if(tennis_shibie)
        {
            this->tennisshibie(cam0, cam1);
            cv::waitKey(1);
            cv::resize(this->img1, this->img1, cv::Size(), 0.5, 0.5);
            cv::resize(this->img2, this->img2, cv::Size(), 0.5, 0.5);
            imgTexture1.Upload(this->img1.data, GL_BGR, GL_UNSIGNED_BYTE);
            imgTexture2.Upload(this->img2.data, GL_BGR, GL_UNSIGNED_BYTE);

            if (mappoint!=cv::Point2f(0,0))
            {
                cv::circle(changdibmp,cv::Point(mappoint.x,mappoint.y),9,cv::Scalar(0,0,255),-1);
                changdibmp.copyTo(org_changdibmp);
            }
            if (mappoint.x==480 && mappoint.y==350)
            {
                cv::circle(changdibmp, cv::Point(mappoint.x + this->point.x / 10, mappoint.y + this->point.y / 10), 9, cv::Scalar(255, 0, 255), -1);
                cv::circle(changdibmp, cv::Point(mappoint.x + this->point_jz.x / 10, mappoint.y + this->point_jz.y / 10), 9, cv::Scalar(100, 150, 200), -1);
            }
            else
            {
                cv::circle(changdibmp, cv::Point(mappoint.x + this->point.y / 10, mappoint.y - this->point.x / 10), 9, cv::Scalar(255, 0, 255), -1);
                cv::circle(changdibmp, cv::Point(mappoint.x + this->point_jz.y / 10, mappoint.y - this->point_jz.x / 10), 9, cv::Scalar(100, 150, 200), -1);
                // std::cout<<"no jz:"<<this->point.x<<' '<<this->point.y<<std::endl;
                // std::cout<<"jz:"<<this->point_jz.x<<' '<<this->point_jz.y<<std::endl;
            }
            //显示场地图
            changdi_map.Activate();
            glColor3f(1.0f, 1.0f, 1.0f);
            changdi.RenderToViewportFlipY();
            changdi.Upload(changdibmp.data, GL_BGR, GL_UNSIGNED_BYTE);
            org_changdibmp.copyTo(changdibmp);
        }

        cv_img_1.Activate();
        glColor3f(1.0f, 1.0f, 1.0f);
        imgTexture1.RenderToViewportFlipY();

        cv_img_2.Activate();
        glColor3f(1.0f, 1.0f, 1.0f);
        imgTexture2.RenderToViewportFlipY();

        d_cam.Activate(s_cam);

        // 绘制坐标系
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
        glColor3f(0.f, 1.0f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
        glEnd();

        //计算记录时间
        if (if_start)
        {
            op_stime = chrono::system_clock::now();
        }
        if (if_end)
        {
            op_time = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - op_stime);
        }

        //调节亮度（伽马值、曝光时间）
        if (if_up)
        {
            float gain = cam0.gain + 1;
            float exposure = cam0.exposureTime + 500;
            cam0.setLight(gain, exposure);
            cam1.setLight(gain, exposure);
            if_up = false;
        }
        if (if_down)
        {
            float gain = cam0.gain - 1;
            float exposure = cam0.exposureTime - 500;
            cam0.setLight(gain, exposure);
            cam1.setLight(gain, exposure);
            if_down = false;
        }

        if(if_start ==false && linshi_p.size()!=0)
        {
            if(arrow_shibie)
            {
                for (int i = linshi_p.size() - 1; i >= max(0, linshi_p.size() - 20); i--)
                {
                    glPointSize(7.0f);
                    glBegin(GL_POINTS);
                    glColor3f(0.0, 0.0, 0.0);
                    glVertex3f((linshi_p[i].x / 100), (linshi_p[i].y / 100), (linshi_p[i].z / 100));
                    glEnd();
                }
            }
            if (tennis_shibie)
            {
                //构建场地三维图
                glLineWidth(3);
                glBegin(GL_LINES);
                glColor3f(255, 0, 255);
                glVertex3f(0, 0, 0);
                glVertex3f(6, 0, 0);
                glColor3f(255, 0, 255);
                glVertex3f(0, 12, 0);
                glVertex3f(6, 12, 0);
                glColor3f(255, 0, 255);
                glVertex3f(0, 0, 0);
                glVertex3f(0, 12, 0);
                glColor3f(255, 0, 255);
                glVertex3f(6, 0, 0);
                glVertex3f(6, 12, 0);
                glEnd();

                for (int i = linshi_p.size() - 2; i >= max(0, linshi_p.size() - 10); i--)
                {
                    if (mappoint.x==480 && mappoint.y==350)
                    {
                        glPointSize(7.0f);
                        glBegin(GL_POINTS);
                        glColor3f(0.0, 0.0, 0.0);
                        std::cout << (mappoint.x + linshi_p[i].x / 10) << ' ' << (mappoint.y + linshi_p[i].y / 10) << std::endl;
                        glVertex3f((6 - (mappoint.x + linshi_p[i].x / 10) / 100), (mappoint.y + linshi_p[i].y / 10) / 100, linshi_p[i].z / 1000);
                        glEnd();
                    }
                    else
                    {
                        glPointSize(7.0f);
                        glBegin(GL_POINTS);
                        glColor3f(0.0, 0.0, 0.0);
                        std::cout << (mappoint.x + linshi_p[i].y / 10) << ' ' << (mappoint.y - linshi_p[i].x / 10) << std::endl;
                        glVertex3f((6 - (mappoint.x + linshi_p[i].y / 10) / 100), (mappoint.y - linshi_p[i].x / 10) / 100, linshi_p[i].z / 1000);
                        glEnd();
                    }
                }

                glPointSize(7.0f);
                glBegin(GL_POINTS);
                glColor3f(0.2, 1.0, 1.0);
                // std::cout<<(mappoint.x+linshi_p[linshi_p.size()-1].y/10)<<' '<<(mappoint.y-linshi_p[linshi_p.size()-1].x/10)<<std::endl;
                glVertex3f((6 - (mappoint.x + linshi_p[linshi_p.size() - 1].x / 10) / 100), (mappoint.y + linshi_p[linshi_p.size() - 1].y / 10) / 100, this->point.z / 1000);
                glEnd();
            }
        }
        else if (if_start == true)
        {
            for (int i = 0; i < this->points.size(); i++)
            {
                glPointSize(3.0f);
                glBegin(GL_POINTS);
                glColor3f(0.0, 0.0, 0.0);
                glVertex3f((this->points[i].x / 100), (this->points[i].y / 100), (this->points[i].z / 100));
                glEnd();
            }
        } 

        //清除所有点
        if (if_clear)
        {
            this->points.clear();
            linshi_p.clear();
            group_l.clear();
            group_r.clear();
            grp_l.clear();
            grp_r.clear();
            record_l.clear();
            record_l.clear();
            if_clear = false;
        }

        //选点
        if (if_xuandian == false)
        {
            cv::Point3f mapresult = get_start_point();
            std::cout<<"map_point:"<<mapresult.x<<' '<<mapresult.y<<std::endl;
            if(mapresult.x==480 && mapresult.y==350)
            {
                std::cout<<"other point"<<std::endl;
            }
            mappoint = cv::Point2f(mapresult.x, mapresult.y);
            if_xuandian = true;
        }

        //帧间时间、帧数
        fps++;
        t_now = time(0);
        auto exat_now = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(exat_now - exat_last);
        //std::cout<<double(duration.count())<<std::endl;
        exat_last = exat_now;
        if (t_now - t0)
        {
            t0 = time(0);
            std::cout << "fps:" << fps << std::endl;
            fps = 0;
        }

        if (if_out)
        {
            for (int i = 0; i < this->points.size(); i++)
            {
                outfile << record_l[i].x << " " << record_l[i].y << " " << record_r[i].x << " " << record_r[i].y << " "
                << this->points[i].x << " " << this->points[i].y << " " << this->points[i].z << double(duration.count()) << '\n';
            }
            outfile << " " << double(op_time.count());
            outfile.close();
            fnamenum++;
            if_clear = true;
            if_out = false;
        }

        pangolin::FinishFrame();
    }
}
