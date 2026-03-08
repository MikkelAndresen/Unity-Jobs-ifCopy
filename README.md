This is a personal hobby project where I look at how one can efficiently copy data conditionally using Unity Job system and their burst compiler.
I started this project in relation to running gpu instancing and wanting an efficient culling solution which added/removed matrices based on whether they are within the camera frustum.
In the end of that project I realized I had a lot of fun just trying to get the culling algortithm to be fast.

When making the culling system I figured out that I could just boil it down to a simple problem, copy all data from one array passing a filter to another.
Or in other words, creating a list based on another array and filter.

I used multiple approaches, at first I started out using the unity IJobParallelForFilter which has since been deprecated.
Due to dissapointing performance of this interface I decided to try my own things essentially making my own version of this.
Since this I worked on this project last the job system has been updated quite a bit and I believe there are builtin interfaces that are now faster than what I managed back then.
I do suspect if I go back to this project I could come up with something faster than what is builtin as well.
