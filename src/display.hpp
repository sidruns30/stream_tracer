/*
    * display.hpp
    *
    *  Code to display the progress on the screen
    * TO DO: add timers in the display
*/
#ifndef DISPLAY_HPP_
    #define DISPLAY_HPP_

    # include <chrono>
    # include <thread>
    # include "global.hpp"
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
        std::string GetLabel() const
        {
            return label;
        }
        void Begin()
        {
            time_begin = std::chrono::high_resolution_clock::now();
            return;
        }
        void End()
        {
            time_end = std::chrono::high_resolution_clock::now();
            time_since_checkpoint = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count();
            total_time_elapsed += time_since_checkpoint;
            return;
        }
        void Reset()
        {
            time_since_checkpoint = 0.;
            return;
        }
        std::string label;
        std::chrono::time_point<std::chrono::high_resolution_clock> time_begin;
        decltype(std::chrono::high_resolution_clock::now()) time_end;
        double total_time_elapsed;
        double time_since_checkpoint;
};

/*
    * Global timer class to store all timers
*/
class Timers
{
    public:
        Timers(std::size_t n_iterations,
               std::size_t display_every) :
               n_iterations(n_iterations),
               display_every (display_every) {};
            
        bool TimerExists(std::string label) const
        {
            for (auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   return true;}
            }
            return false;
        }

        void AddTimer(std::string label)
        {   
            if (!TimerExists(label))
            {   timers.push_back(Timer(label)); }
        }

        void AddTimer(std::vector<std::string> labels)
        {
            for (auto& label : labels)
            {   
                if (!TimerExists(label))
                {   timers.push_back(Timer(label)); }
            }
        }

        void BeginTimer(std::string label)
        {
            for (auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   timer.Begin(); }
            }
        }

        void EndTimer(std::string label)
        {
            for (auto& timer : timers)
            {
                if (timer.GetLabel() == label)
                {   timer.End(); }
            }
        }

        bool PrintTimers(std::size_t current_iteration)
        {
            if (current_iteration % display_every != 0)
            {   return false;}

            double total = 0.;
            double total_checkpoint = 0.;
            for (auto& timer : timers)
            {
                total += timer.total_time_elapsed;
                total_checkpoint += timer.time_since_checkpoint;
            }
            std::cout << colors::red << "Iteration: " << colors::yellow << 
            "[" << current_iteration << "/" << n_iterations << 
            "]" << colors::reset << std::endl;

            for (auto& timer : timers)
            {
                std::cout << colors::blue << timer.GetLabel() << colors::reset << 
                "[% total]: " << colors::green << 100. * timer.total_time_elapsed / total << "% \t"
                << colors::reset << "[time]: " << colors::yellow <<
                timer.total_time_elapsed * 1.e-9 << " seconds" << std::endl;
                timer.Reset();
            }
            std::cout << colors::red << "Time elapsed: " << colors::green << total * 1.e-9 << " seconds" << colors::red;
            std::cout << "\t \t Time left: " << colors::green << (total * (n_iterations - current_iteration) / current_iteration) * 1.e-9 
            << " seconds" << colors::reset << std::endl;
            std::cout << std::endl;
            return true;
        }

    private:
        std::vector<Timer> timers;
        std::size_t n_iterations;
        std::size_t display_every;
};



#endif