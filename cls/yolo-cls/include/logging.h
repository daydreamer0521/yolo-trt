/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include "NvInferRuntimeCommon.h"
#include "macros.h"

using Severity = nvinfer1::ILogger::Severity;

class LogStreamConsumerBuffer : public std::stringbuf {
   public:
    LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mOutput(stream), mPrefix(prefix), mShouldLog(shouldLog) {}

    LogStreamConsumerBuffer(LogStreamConsumerBuffer&& other) : mOutput(other.mOutput) {}

    ~LogStreamConsumerBuffer() {
        // std::streambuf::pbase() 返回指向输出序列中已缓冲部分起始位置的指针
        // std::streambuf::pptr() 返回指向输出序列当前写入位置的指针
        // 如果起始位置指针与当前写入位置指针不相等，则调用 putOutput() 将缓冲区内容写入流
        if (pbase() != pptr()) {
            putOutput();
        }
    }

    // 将流缓冲区同步到对应输出流，成功时返回 0
    // 同步操作包括：将缓冲区内容写入流、重置缓冲区并刷新输出流
    virtual int sync() {
        putOutput();
        return 0;
    }

    void putOutput() {
        if (mShouldLog) {
            // 在输出前添加时间戳前缀
            std::time_t timestamp = std::time(nullptr);
            tm* tm_local = std::localtime(&timestamp);
            std::cout << "[";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            // std::stringbuf::str() 获取缓冲区中的字符串内容
            // 将带有前缀的缓冲区内容写入输出流
            mOutput << mPrefix << str();
            // 清空缓冲区
            str("");
            // 刷新输出流
            mOutput.flush();
        }
    }

    void setShouldLog(bool shouldLog) { mShouldLog = shouldLog; }

   private:
    std::ostream& mOutput;
    std::string mPrefix;
    bool mShouldLog;
};

//!
//! \class LogStreamConsumerBase
//! \brief 用于在 LogStreamConsumer 中在 std::ostream 之前初始化 LogStreamConsumerBuffer 的辅助类
//!
class LogStreamConsumerBase {
   public:
    LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog) {}

   protected:
    LogStreamConsumerBuffer mBuffer;
};

//!
//! \class LogStreamConsumer
//! \brief 辅助对象：在记录日志时使用 C++ 流语法的便利封装。
//!  基类顺序为 LogStreamConsumerBase 然后是 std::ostream。
//!  之所以如此，是因为需要先用 LogStreamConsumerBase 初始化 LogStreamConsumerBuffer 成员，
//!  然后再将该缓冲区地址传递给 std::ostream，避免将未初始化的缓冲区地址传入 std::ostream。
//!  请不要更改父类的继承顺序。
//!
class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream {
   public:
    //! \brief 创建一个用于记录指定级别消息的 LogStreamConsumer。
    //!  报告级别（reportable severity）决定消息是否达到记录阈值。
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity),
          std::ostream(&mBuffer)  // 将流缓冲区与 std::ostream 关联
          ,
          mShouldLog(severity <= reportableSeverity),
          mSeverity(severity) {}

    LogStreamConsumer(LogStreamConsumer&& other)
        : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog),
          std::ostream(&mBuffer)  // links the stream buffer with the stream
          ,
          mShouldLog(other.mShouldLog),
          mSeverity(other.mSeverity) {}

    void setReportableSeverity(Severity reportableSeverity) {
        mShouldLog = mSeverity <= reportableSeverity;
        mBuffer.setShouldLog(mShouldLog);
    }

   private:
    static std::ostream& severityOstream(Severity severity) {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static std::string severityPrefix(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                return "[F] ";
            case Severity::kERROR:
                return "[E] ";
            case Severity::kWARNING:
                return "[W] ";
            case Severity::kINFO:
                return "[I] ";
            case Severity::kVERBOSE:
                return "[V] ";
            default:
                assert(0);
                return "";
        }
    }

    bool mShouldLog;
    Severity mSeverity;
};

    //! \class Logger
    //!
    //! \brief 管理 TensorRT 工具与示例程序日志的类
    //!
    //! \details 本类为 TensorRT 工具与示例提供统一的控制台日志接口，
    //! 支持两类日志消息：
    //!
    //! - 带严重度的调试消息（info、warning、error、internal error/fatal）
    //! - 测试通过/失败的结果消息
    //!
    //! 让所有示例使用本类进行日志记录而不是直接写入 stdout/stderr 的优点是：
    //! 控制日志详细程度和格式化输出的逻辑被集中管理，便于维护和统一输出样式。
    //!
    //! 将来本类可以扩展以支持将测试结果以标准格式（例如 JUnit XML）导出到文件，
    //! 并提供额外元数据（例如测试运行时长）。
    //!
    //! TODO: 为了兼容已有示例，本类目前直接继承自 nvinfer1::ILogger 接口，
    //! 这在设计上不够清晰，因为无法明确区分来自 TensorRT 库的消息与示例程序自身的消息。
    //!
    //! 将来（当所有示例改为使用 Logger::getTRTLogger() 注册 ILogger 时）我们可以重构本类，
    //! 取消继承并将 nvinfer1::ILogger 的实现作为 Logger 的成员。

class Logger : public nvinfer1::ILogger {
   public:
    Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}

    //!
    //! \enum TestResult
    //! \brief 表示某个测试的状态
    //!
    enum class TestResult {
        kRUNNING,  //!< 测试正在运行
        kPASSED,   //!< 测试通过
        kFAILED,   //!< 测试失败
        kWAIVED    //!< 测试被放弃（waived）
    };

    //!
    //! \brief 向前兼容的方法：获取与此 Logger 关联的 nvinfer::ILogger
    //! \return 与此 Logger 关联的 nvinfer1::ILogger 引用
    //!
    //! TODO 当所有示例都更新为使用此方法向 TensorRT 注册 logger 后，
    //! 我们可以取消 Logger 从 ILogger 的继承关系
    //!
    nvinfer1::ILogger& getTRTLogger() { return *this; }

    //!
    //! \brief nvinfer1::ILogger::log() 虚方法的实现
    //!
    //! 注意：示例程序不应直接调用此函数；一旦我们取消对 nvinfer1::ILogger 的继承，此方法将被移除
    //!
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }

    //!
    //! \brief 控制日志输出详尽程度的方法
    //!
    //! \param severity 日志器将只输出严重度不低于该级别的消息。
    //!
    void setReportableSeverity(Severity severity) { mReportableSeverity = severity; }

    //!
    //! \brief 用于保存特定测试的日志信息的不透明句柄
    //!
    //! 该对象是一个不透明句柄，保存了 Logger 在打印测试结果时所需的信息。
    //! 示例程序必须调用 Logger::defineTest() 来获取一个 TestAtom，该对象可用于
    //! Logger::reportTest{Start,End}()。
    //!
    class TestAtom {
       public:
        TestAtom(TestAtom&&) = default;

       private:
        friend class Logger;

        TestAtom(bool started, const std::string& name, const std::string& cmdline)
            : mStarted(started), mName(name), mCmdline(cmdline) {}

        bool mStarted;
        std::string mName;
        std::string mCmdline;
    };

    //!
    //! \brief 定义一个用于日志记录的测试
    //!
    //! \param[in] name 测试的名称。该名称应该以 "TensorRT" 开头，并由点分隔的字符串组成，
    //!                  仅包含字符 [A-Za-z0-9_]。
    //!                  例如："TensorRT.sample_googlenet"
    //! \param[in] cmdline 用于重现测试的命令行字符串
    //
    //! \return 可用于 Logger::reportTest{Start,End}() 的 TestAtom 对象。
    //!
    static TestAtom defineTest(const std::string& name, const std::string& cmdline) {
        return TestAtom(false, name, cmdline);
    }

    //!
    //! \brief A convenience overloaded version of defineTest() that accepts an array of command-line arguments
    //!        as input
    //!
    //! \param[in] name 测试名称
    //! \param[in] argc 命令行参数个数
    //! \param[in] argv 命令行参数数组（以 C 字符串形式给出）
    //!
    //! \return 可用于 Logger::reportTest{Start,End}() 的 TestAtom 对象。
    static TestAtom defineTest(const std::string& name, int argc, char const* const* argv) {
        auto cmdline = genCmdlineString(argc, argv);
        return defineTest(name, cmdline);
    }

    //!
    //! \brief Report that a test has started.
    //!
    //! \pre reportTestStart() has not been called yet for the given testAtom
    //!
    //! \param[in] testAtom The handle to the test that has started
    //!
    static void reportTestStart(TestAtom& testAtom) {
        reportTestResult(testAtom, TestResult::kRUNNING);
        assert(!testAtom.mStarted);
        testAtom.mStarted = true;
    }

    //!
    //! \brief Report that a test has ended.
    //!
    //! \pre reportTestStart() has been called for the given testAtom
    //!
    //! \param[in] testAtom The handle to the test that has ended
    //! \param[in] result The result of the test. Should be one of TestResult::kPASSED,
    //!                   TestResult::kFAILED, TestResult::kWAIVED
    //!
    static void reportTestEnd(const TestAtom& testAtom, TestResult result) {
        assert(result != TestResult::kRUNNING);
        assert(testAtom.mStarted);
        reportTestResult(testAtom, result);
    }

    static int reportPass(const TestAtom& testAtom) {
        reportTestEnd(testAtom, TestResult::kPASSED);
        return EXIT_SUCCESS;
    }

    static int reportFail(const TestAtom& testAtom) {
        reportTestEnd(testAtom, TestResult::kFAILED);
        return EXIT_FAILURE;
    }

    static int reportWaive(const TestAtom& testAtom) {
        reportTestEnd(testAtom, TestResult::kWAIVED);
        return EXIT_SUCCESS;
    }

    static int reportTest(const TestAtom& testAtom, bool pass) {
        return pass ? reportPass(testAtom) : reportFail(testAtom);
    }

    Severity getReportableSeverity() const { return mReportableSeverity; }

   private:
    //!
    //! \brief 返回用于在日志消息前加上严重度前缀的字符串
    //!
    static const char* severityPrefix(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                return "[F] ";
            case Severity::kERROR:
                return "[E] ";
            case Severity::kWARNING:
                return "[W] ";
            case Severity::kINFO:
                return "[I] ";
            case Severity::kVERBOSE:
                return "[V] ";
            default:
                assert(0);
                return "";
        }
    }

    //!
    //! \brief 返回用于在测试结果消息前加上结果前缀的字符串
    //!
    static const char* testResultString(TestResult result) {
        switch (result) {
            case TestResult::kRUNNING:
                return "RUNNING";
            case TestResult::kPASSED:
                return "PASSED";
            case TestResult::kFAILED:
                return "FAILED";
            case TestResult::kWAIVED:
                return "WAIVED";
            default:
                assert(0);
                return "";
        }
    }

    //!
    //! \brief 根据给定的严重度返回合适的输出流（cout 或 cerr）
    //!
    static std::ostream& severityOstream(Severity severity) {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    //!
    //! \brief 实现测试结果日志打印的方法
    //!
    static void reportTestResult(const TestAtom& testAtom, TestResult result) {
        severityOstream(Severity::kINFO) << "&&&& " << testResultString(result) << " " << testAtom.mName << " # "
                                         << testAtom.mCmdline << std::endl;
    }

    //!
    //! \brief 根据给定的 (argc, argv) 生成命令行字符串
    //!
    static std::string genCmdlineString(int argc, char const* const* argv) {
        std::stringstream ss;
        for (int i = 0; i < argc; i++) {
            if (i > 0)
                ss << " ";
            ss << argv[i];
        }
        return ss.str();
    }

    Severity mReportableSeverity;
};

namespace {

//!
//! \brief 生成一个 LogStreamConsumer 对象，用于记录严重度为 kVERBOSE 的日志
//!
//! 示例用法:
//!
//!     LOG_VERBOSE(logger) << "hello world" << std::endl;
//!
inline LogStreamConsumer LOG_VERBOSE(const Logger& logger) {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
}

//!
//! \brief 生成一个 LogStreamConsumer 对象，用于记录严重度为 kINFO 的日志
//!
//! 示例用法:
//!
//!     LOG_INFO(logger) << "hello world" << std::endl;
//!
inline LogStreamConsumer LOG_INFO(const Logger& logger) {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
}

//!
//! \brief 生成一个 LogStreamConsumer 对象，用于记录严重度为 kWARNING 的日志
//!
//! 示例用法:
//!
//!     LOG_WARN(logger) << "hello world" << std::endl;
//!
inline LogStreamConsumer LOG_WARN(const Logger& logger) {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
}

//!
//! \brief 生成一个 LogStreamConsumer 对象，用于记录严重度为 kERROR 的日志
//!
//! 示例用法:
//!
//!     LOG_ERROR(logger) << "hello world" << std::endl;
//!
inline LogStreamConsumer LOG_ERROR(const Logger& logger) {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
}

//!
//! \brief 生成一个 LogStreamConsumer 对象，用于记录严重度为 kINTERNAL_ERROR
//!         （相当于“致命”级别）
//!
//! 示例用法:
//!
//!     LOG_FATAL(logger) << "hello world" << std::endl;
//!
inline LogStreamConsumer LOG_FATAL(const Logger& logger) {
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
}

}  // anonymous namespace

#endif  // TENSORRT_LOGGING_H
