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

    

    template <typename T>
    class DisplayTerminal
    {   
        public:
            DisplayTerminal(T current_progresss,
                            T total_progress,
                            T progress_threshold) : 
                            current_progress(current_progress),
                            total_progress(total_progress),
                            progress_threshold(progress_threshold);
            void UpdateProgress(T progress_increment)
            void DisplayProgress();
            ~DisplayTerminal() = default;
        private:
            T current_progress;
            T total_progress;
            T progress_increment;
            T progress_threshold;
    };

#endif