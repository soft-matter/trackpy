-- MySQL dump 10.13  Distrib 5.5.24, for debian-linux-gnu (i686)
--
-- Host: localhost    Database: exp3
-- ------------------------------------------------------
-- Server version	5.5.24-0ubuntu0.12.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Batch`
--

DROP TABLE IF EXISTS `Batch`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Batch` (
  `batch` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `trial` smallint(5) unsigned DEFAULT NULL,
  `stack` smallint(5) unsigned DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `comment` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`batch`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Features`
--

DROP TABLE IF EXISTS `Features`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Features` (
  `trial` smallint(4) unsigned NOT NULL DEFAULT '0',
  `stack` smallint(2) unsigned NOT NULL DEFAULT '0',
  `version` tinyint(2) unsigned NOT NULL DEFAULT '0',
  `frame` int(6) unsigned zerofill NOT NULL,
  `feature` smallint(3) unsigned NOT NULL AUTO_INCREMENT,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `mass` float NOT NULL,
  `size` float NOT NULL,
  `ecc` float NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`frame`,`feature`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `MSD`
--

DROP TABLE IF EXISTS `MSD`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `MSD` (
  `stack` int(10) unsigned NOT NULL DEFAULT '0',
  `trial` int(10) unsigned NOT NULL DEFAULT '0',
  `probe` int(10) unsigned NOT NULL,
  `datapoint_num` mediumint(8) unsigned NOT NULL,
  `t` float NOT NULL,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `x2` float NOT NULL,
  `y2` float NOT NULL,
  `r2` float NOT NULL,
  `N` int(10) unsigned NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`probe`,`datapoint_num`)
) ENGINE=MyISAM AUTO_INCREMENT=4120 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Probes`
--

DROP TABLE IF EXISTS `Probes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Probes` (
  `trial` smallint(5) unsigned NOT NULL,
  `stack` smallint(5) unsigned NOT NULL,
  `version` tinyint(3) unsigned NOT NULL,
  `probe` int(10) unsigned NOT NULL,
  `fit_A` float DEFAULT NULL,
  `fit_n` float DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`probe`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Stacks`
--

DROP TABLE IF EXISTS `Stacks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Stacks` (
  `trial` smallint(4) unsigned NOT NULL,
  `stack` smallint(2) unsigned NOT NULL AUTO_INCREMENT,
  `start` time DEFAULT NULL,
  `end` time DEFAULT NULL,
  `vstart` time DEFAULT NULL,
  `vduration` time DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` enum('reserved','muxing started','muxed') DEFAULT NULL,
  `video` varchar(31) DEFAULT NULL,
  `comment` varchar(1023) DEFAULT NULL,
  PRIMARY KEY (`trial`,`stack`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Trajectories`
--

DROP TABLE IF EXISTS `Trajectories`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Trajectories` (
  `trial` smallint(5) unsigned NOT NULL,
  `stack` smallint(5) unsigned NOT NULL,
  `version` tinyint(3) unsigned NOT NULL,
  `probe` int(10) unsigned NOT NULL,
  `frame` int(10) unsigned zerofill NOT NULL,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `mass` float NOT NULL,
  `size` float NOT NULL,
  `ecc` float NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`probe`,`frame`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Trials`
--

DROP TABLE IF EXISTS `Trials`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Trials` (
  `trial` smallint(4) unsigned NOT NULL AUTO_INCREMENT,
  `description` varchar(63) DEFAULT NULL,
  `who` varchar(63) DEFAULT 'Dan Allan',
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `probes` varchar(1023) DEFAULT NULL,
  `comment` varchar(1023) DEFAULT NULL,
  PRIMARY KEY (`trial`)
) ENGINE=InnoDB AUTO_INCREMENT=60 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2012-09-05 16:40:04
