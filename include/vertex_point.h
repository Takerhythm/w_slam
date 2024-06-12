#ifndef SLAM_BACKEND_POINTVERTEX_H
#define SLAM_BACKEND_POINTVERTEX_H

#include "vertex.h"

namespace slam {
namespace backend {

/**
 * @brief 以xyz形式参数化的顶点
 */
class VertexPoint : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() : Vertex(3) {}

    std::string TypeInfo() const { return "VertexPoint"; }
};

}
}

#endif
