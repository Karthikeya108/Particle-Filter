/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	
	double gps_x = x;
	double gps_y = y;
	
	default_random_engine gen;
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(gps_x, std_x);

	// TODO: Create normal distributions for y and theta
	normal_distribution<double> dist_y(gps_y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

        for (int k=0; k < num_particles; k++) {
	  Particle particle;
	  particle.id = k;
	  particle.x = dist_x(gen);
	  particle.y = dist_y(gen);
	  particle.theta = dist_theta(gen);
	  particle.weight = 1.0;

	  particles.push_back(particle);
	  weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
        
	default_random_engine gen;
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];        

        // Random Gaussian Noise distributions
	normal_distribution<double> noise_x(0.0, std_x);
	normal_distribution<double> noise_y(0.0, std_y);
	normal_distribution<double> noise_theta(0.0, std_theta);
	
	for (int k=0; k < num_particles; k++) {
	  Particle& particle = particles[k];
	  // if yaw_rate is non zero
          if(fabs(yaw_rate) > 0.0001) {
            
            particle.x += velocity/yaw_rate * ( sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta) );
	    particle.y += velocity/yaw_rate * ( cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t) );
	    particle.theta +=  yaw_rate * delta_t;

	  } else {
             particle.x += velocity * delta_t * cos(particle.theta);
	     particle.y += velocity * delta_t * sin(particle.theta);
	  }

	  particle.x += noise_x(gen);
	  particle.y += noise_y(gen);
	  particle.theta += noise_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int k=0; k < observations.size(); k++) {
	  // Current observation
	  LandmarkObs obs = observations[k];
	  double min_dist = -1.0;
	  for (int l=0; l < predicted.size(); l++) {
	    // Current prediction
            LandmarkObs pred = predicted[l];
	    double distance = dist(obs.x, obs.y, pred.x, pred.y);
	    if (min_dist < 0 || distance < min_dist) {
	        min_dist = distance;
		observations[k].id = l;
	    }
	  }
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double lm_std_x = std_landmark[0];
	double lm_std_y = std_landmark[1];

	for (int k=0; k < num_particles; k++) {
	  Particle& particle = particles[k];
	  vector<LandmarkObs> lm_in_range;
	  for(int l=0; l < map_landmarks.landmark_list.size(); l++) {
	    int lm_id = map_landmarks.landmark_list[l].id_i;
	    double lm_x = map_landmarks.landmark_list[l].x_f;
	    double lm_y = map_landmarks.landmark_list[l].y_f;
            
	    double distance = dist(particle.x, particle.y, lm_x, lm_y);

	    if(distance < sensor_range) {
	      LandmarkObs lmObs;
	      lmObs.id = lm_id;
	      lmObs.x = lm_x;
	      lmObs.y = lm_y;
	      
	      lm_in_range.push_back(lmObs);  
	    }
	  }

          // Convert observation coordinates to map coordinates
          vector<LandmarkObs> mapObs;
	  for (int l=0; l < observations.size(); l++) {
	    LandmarkObs o = observations[l];
	    LandmarkObs m;
	    m.x = particle.x + cos(particle.theta) * o.x - sin(particle.theta) * o.y;
	    m.y = particle.y + sin(particle.theta) * o.x + cos(particle.theta) * o.y;
	    mapObs.push_back(m);
	  }
         
	  dataAssociation(lm_in_range, mapObs);
          
	  particle.weight = 1.0;
	  vector<int> associations;
          vector<double> sense_x;
          vector<double> sense_y;

	  for (int l=0; l < mapObs.size(); l++) {
	    LandmarkObs obs = mapObs[l];
	    LandmarkObs lm = lm_in_range[obs.id];
 
            associations.push_back(lm.id);
	    sense_x.push_back(obs.x);
	    sense_y.push_back(obs.y);

            double norm = 1 / (2 * M_PI * lm_std_x * lm_std_y);
            double term1 = pow(obs.x - lm.x, 2) / (2 * lm_std_x * lm_std_x);
	    double term2 = pow(obs.y - lm.y, 2) / (2 * lm_std_y * lm_std_y);

	    particle.weight *= norm * exp(- (term1 + term2));
	  }
	  particle = SetAssociations(particle, associations, sense_x, sense_y);
          weights[k] = particle.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> samples;
	for (int k=0; k < num_particles; k++) {
	  samples.push_back(particles[distribution(gen)]);
	}
	particles = samples;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
