#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>
#include "kernels/ellipse_fitting.cu"

auto main(int argc, char* argv[]) -> int
{
    argparse::ArgumentParser program("polarization");
    argparse::ArgumentParser ellipses_command("ellipses");
    ellipses_command.add_argument("I0").required();
    ellipses_command.add_argument("I45").required();
    ellipses_command.add_argument("I90").required();
    ellipses_command.add_argument("I135").required();

    program.add_subparser(ellipses_command);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (!program.is_subcommand_used(ellipses_command)) {
        return 1;
    }

    std::string I0_path = ellipses_command.get<std::string>("I0");
    std::string I45_path = ellipses_command.get<std::string>("I45");
    std::string I90_path = ellipses_command.get<std::string>("I90");
    std::string I135_path = ellipses_command.get<std::string>("I135");

    cv::Mat I0 = cv::imread(I0_path, cv::IMREAD_GRAYSCALE);
    cv::Mat I45 = cv::imread(I45_path, cv::IMREAD_GRAYSCALE);
    cv::Mat I90 = cv::imread(I90_path, cv::IMREAD_GRAYSCALE);
    cv::Mat I135 = cv::imread(I135_path, cv::IMREAD_GRAYSCALE);

    if (I0.empty() || I45.empty() || I90.empty() || I135.empty())
        return 1;

    return 0;
}