/*
    * display.hpp
    *
    *  Code to display the progress on the screen
    * TO DO: add timers in the display
*/
#ifndef DISPLAY_HPP_
    #define DISPLAY_HPP_

    # include <iostream>
    # include <chrono>
    # include <thread>

    namespace colors
    {
        const std::string red = "\033[1;31m";
        const std::string green = "\033[1;32m";
        const std::string yellow = "\033[1;33m";
        const std::string blue = "\033[1;34m";
        const std::string magenta = "\033[1;35m";
        const std::string cyan = "\033[1;36m";
        const std::string white = "\033[1;37m";
        const std::string reset = "\033[0m";
    }

    struct Timer
{
        Timer(const std::string& label) :   time_begin(std::chrono::high_resolution_clock::now()), 
                                            label(label), total_time_elapsed(0.), 
                                            time_since_checkpoint(0.) {}
        void Reset()
        {
            time_begin = std::chrono::high_resolution_clock::now();
            total_time_elapsed = 0.;
            time_since_checkpoint = 0.;
            return;
        }
        void Checkpoint()
        {
            auto current_time = std::chrono::high_resolution_clock::now();
            time_since_checkpoint = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - elapsed_checkpoint).count();
            total_time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - time_begin).count();
            elapsed_checkpoint = std::chrono::high_resolution_clock::now();
            return;
        }
        std::string GetLabel() const
        {
            return label;
        }
        double GetTotalTimeElapsed() const
        {
            return total_time_elapsed;
        }
        double GetTimeSinceCheckpoint() const
        {
            return time_since_checkpoint;
        }
        std::string label;
        std::chrono::time_point<std::chrono::high_resolution_clock> time_begin;
        decltype(std::chrono::high_resolution_clock::now()) elapsed_checkpoint;
        double total_time_elapsed;
        double time_since_checkpoint;
};

/*
    * Global timer class to store all timers
*/
class Timers
{
    public:
        Timers(std::size_t current_iteration,
               std::size_t n_iterations) :
               current_iteration(current_iteration),
               n_iterations(n_iterations) {};
        void AddTimer(const Timer& timer)
        {
            timers.push_back(timer);
        }
        void Checkpoint()
        {
            for (auto& timer : timers)
            {
                timer.Checkpoint();
            }
        }

        void PrintTimers()
        {
            double total = 0.;
            for (auto& timer : timers)
            {
                total += timer.total_time_elapsed;
            }
            for (auto& timer : timers)
            {
                std::cout << colors::blue << timer.GetLabel() << colors::reset << " fraction: "
                          << colors::green << 100*timer.total_time_elapsed/total << colors::reset << std::endl;
            }
            std::cout << colors::blue << "Total time elapsed: " << colors::reset << total << std::endl;
        }


    private:
        std::vector<Timer> timers;
        std::size_t current_iteration;
        std::size_t n_iterations;
};

    template <typename T>
    class DisplayTerminal
    {   
        public:
            DisplayTerminal(T complete_progress,
                            std::string taskName,
                            T progress_threshold) : 
                            current_progress(0),
                            complete_progress(complete_progress),
                            taskName(taskName),
                            progress_threshold(progress_threshold),
                            last_displayed_progress (0.),
                            start_time(std::chrono::high_resolution_clock::now()) {};
            DisplayTerminal(T complete_progress,
                            std::string taskName): 
                            current_progress(0),
                            complete_progress(complete_progress),
                            taskName(taskName),
                            progress_threshold(complete_progress / 20.),
                            last_displayed_progress (0.),
                            start_time(std::chrono::high_resolution_clock::now()) {};

            void UpdateProgress(T progress_increment);
            void DisplayProgress();
            ~DisplayTerminal() = default;
        private:
            T current_progress;
            T complete_progress;
            T progress_increment;
            T progress_threshold;
            T last_displayed_progress;
            decltype(std::chrono::high_resolution_clock::now()) start_time;
            std::string taskName;
    };;

    template <typename T>
    void DisplayTerminal<T>::UpdateProgress(T progress_increment)
    {
        current_progress += progress_increment;
        return;
    }

    /*
        Display the progress on the terminal via *
        An extra * is added every time the progress exceeds the threshold
        <Task Name>: Expected finish time: %d seconds
        ******************-------------------------- [N% Complete]
                         ^ Current Progress        ^ Complete Progress
    */
    template <typename T>
    void DisplayTerminal<T>::DisplayProgress()
    {
        if (current_progress - last_displayed_progress < progress_threshold)
        {   return;}
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
        
        // Print start time
        auto expected_time = static_cast<float> ( 1.e-9 * elapsed_time * (complete_progress / current_progress));
        auto time_remaining = expected_time - 1.e-9 * elapsed_time;

        T progress = current_progress / complete_progress;
        auto n_chars = complete_progress / progress_threshold;
        auto n_chars_star = current_progress / progress_threshold;
        auto n_chars_dash = n_chars - n_chars_star;
        auto progress_percent = progress * 100;
        std::cout << colors::blue << taskName << colors::reset << ": Expected time remaining: " << time_remaining << " seconds" << std::endl;
        std::cout << colors::green << "Progress: " << colors::reset;
        std::cout << "[";
        std::cout << colors::yellow;
        for (auto i=0; i<n_chars_star; i++)
        {   std::cout << "*";   }
        std::cout << colors::red;
        for (auto i=0; i<n_chars_dash; i++)
        {   std::cout << "-";   }
        std::cout << colors::reset;
        std::cout << "] \t (" << progress_percent << "% Complete)" << std::endl;
        last_displayed_progress = current_progress;
        return;
    }

#endif