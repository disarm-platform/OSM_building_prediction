## Helper functions for predicting residential status of OSM buildings
library(rgeos)
library(RANN)

get_data <- function(structures_poly, training=TRUE, impervious=TRUE, cropped_by=NULL){
  
  # Subset only those with type
  with_type <- which(!is.na(structures_poly$type))
  
  if(training==TRUE){
    structures_poly_selected <- structures_poly[with_type,]
  }else{
    structures_poly_selected <- structures_poly[-with_type,]  
  }
  
  # Simplify structure polygons to points
  structure_points <- SpatialPointsDataFrame(gCentroid(structures_poly_selected,byid=TRUE), 
                                             structures_poly_selected@data, match.ID=FALSE)
  
  # Calculate distance to road (nearest road coordinate)
  get_coords <- function(x){x@Lines}
  road_coords <- sapply(roads@lines, get_coords)
  road_coords <- sapply(road_coords, coordinates)
  road_coords_all <- do.call("rbind", road_coords)
  
  structure_points$dist_to_nearest_road <- nn2(road_coords_all, structure_points@coords, eps=0,searchtype="priority", k=2)$nn.dists[,2]
  
  # Calculate areas
  areas <- sapply(slot(structures_poly_selected, "polygons"), function(x) sapply(slot(x, "Polygons"), slot, "area"))
  structure_points$poly_area <- sapply(areas, max) #Some buildings have multiple polygons so choose largest
  structure_points$n_poly <- sapply(areas, length)
  
  # Calculate polygon 'complexity'
  poly_coords <- sapply(slot(structures_poly_selected, "polygons"), function(x) sapply(slot(x, "Polygons"), slot, "coords"))
  structure_points$poly_complexity <- sapply(poly_coords, function(x){length(unlist(x))})
  
  # Calculate distance to nearest structure
  structure_points$dist_to_nearest <- nn2(structure_points@coords,structure_points@coords, eps=0,searchtype="priority", k=5)$nn.dists[,2]
  
  # Calculate features of nearest structure
  structure_points$area_of_nearest <- structure_points$poly_area[nn2(structure_points@coords,structure_points@coords, eps=0,searchtype="priority", k=2)$nn.idx[,2]]
  structure_points$poly_complexity_of_nearest <- structure_points$poly_complexity[nn2(structure_points@coords,structure_points@coords, eps=0,searchtype="priority", k=2)$nn.idx[,2]]
  structure_points$n_poly_of_nearest <- structure_points$n_poly[nn2(structure_points@coords,structure_points@coords, eps=0,searchtype="priority", k=2)$nn.idx[,2]]
  
  # Import data on 'imperviousness' extracted from Midekisa et al. 2017 PloS ONE 12(9) e0184926
  Impervious <- read.csv(paste0(country_code,"_raw_OSM_062617.csv"))
  Impervious_buffer <- read.csv(paste0(country_code,"_resample_OSM_062617.csv"))
    
  # Merge
  structure_points <- merge(structure_points, Impervious[,c("OSM_ID", "Impervious")], by.x="osm_id",by.y="OSM_ID")
  structure_points <- merge(structure_points, Impervious_buffer[,c("OSM_ID", "Impervious_buffer")], by.x="osm_id",by.y="OSM_ID")
  structure_points$Impervious_of_nearest <- structure_points$Impervious[nn2(structure_points@coords,structure_points@coords, eps=0,searchtype="priority", k=2)$nn.idx[,2]]

  # Add lat long as columns
  structure_points$lng <- structure_points@coords[,1]
  structure_points$lat <- structure_points@coords[,2]
  
  return(structure_points)
}

## Identify optimal cutoff for producing equal sensitivity and specificity
optimal_cutoff <- function(predictions, observations){
  
  perc_corr_class_sprayable <- NULL
  perc_corr_class_not_sprayable <- NULL
  
  true_num_sprayable <- sum(observations)
  true_num_not_sprayable <- sum(observations==0)
  
  for(i in 1:1000){
    predicted_class_loop <- ifelse(predictions>=i/1000,1,0)
    perc_corr_class_sprayable <- c(perc_corr_class_sprayable, 
                                   round(sum(observations==1 & predicted_class_loop==1,na.rm=T)/true_num_sprayable,3))
    
    perc_corr_class_not_sprayable <- c(perc_corr_class_not_sprayable, 
                                       round(sum(observations==0 & predicted_class_loop==0,na.rm=T)/true_num_not_sprayable,3))
  }
  
  plot_data <- data.frame(threshold = rep(seq(0,1,length.out=1000),2),
                          perc_corr_class = c(perc_corr_class_sprayable, perc_corr_class_not_sprayable),
                          group = c(rep("Sprayable",1000), rep("Not sprayable",1000)))
  
  ggplot(data=plot_data, aes(x=threshold, y=perc_corr_class, group=group, color=group)) + 
    geom_line(size=1.5) + xlab("Threshold") + ylab("Proportion correctly classified")
  
  # Identify where the lines overlap
  #opt_threshold <- (which(perc_corr_class_sprayable == perc_corr_class_not_sprayable)/1000)[1]
  opt_threshold <- which.min((perc_corr_class_sprayable - perc_corr_class_not_sprayable)^2)[1]/1000
  
  
  # Table predicted class against observed using different thresholds
  predicted_class <- ifelse(predictions>=opt_threshold,1,0) # Threshold of 0.7 gives 95% sensitivity in both
  table(predicted_class, observations)
  return(list(plot_data = plot_data, 
              opt_cutoff = opt_threshold,
              confusion_matrix = table(predicted_class, observations)))
}