import os.path

from skimage.feature import match_descriptors, SIFT, plot_matches
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import csv
import rasterio as rt
import cv2
import kornia as K
import kornia.feature as KF
import torch as th
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *
from pyproj import CRS, Transformer
from numpy import floor
import time
import traceback

device = K.utils.get_cuda_or_mps_device_if_available()
# device = "cpu"
def xy_to_map(x, y, x0, y0, xres, yres, offset_x, offset_y):
    X = x0 + (offset_x + x)*xres
    Y = y0 + (offset_y + y)*yres
    return X, Y

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = KF.get_laf_center(kp1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
    mkpts2 = KF.get_laf_center(kp2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
    return mkpts1, mkpts2

def wgs_to_utm(lon, lat):
    zone = int(floor(((180 + lon) / 6) % 60) + 1)
    is_northern = 'south' if lat < 0 else 'north'

    to_crs = CRS.from_proj4("+proj=utm +zone="+str(zone)+' +'+is_northern+' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    from_crs = CRS.from_epsg(4326)

    proj = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return proj.transform(lon, lat)


loc_1 = (r'D:\e04\mrs\gloc\inp\New folder\22_70_HH.tif',r'D:\e04\mrs\gloc\inp\New folder\22_70_HV.tif', r'D:\e04\mrs\gloc\inp\New folder\22_71_HH.tif',
         r'D:\e04\mrs\gloc\inp\New folder\22_71_HV.tif')
loc_2 = (r'D:\e04\mrs\gloc\inp\New folder\22_70_ref.tif',r'D:\e04\mrs\gloc\inp\New folder\22_70_ref.tif', r'D:\e04\mrs\gloc\inp\New folder\22_71_ref.tif',
         r'D:\e04\mrs\gloc\inp\New folder\22_71_ref.tif',
         r'D:\e04\mrs\gloc\inp\246994651\scene_VH\ref.tif',r'D:\e04\mrs\gloc\inp\246994651\scene_VV\ref.tif')
out_ = ('D:\\e04\\mrs\\gloc\\out_auto\\new\\22_70_hh','D:\\e04\\mrs\\gloc\\out_auto\\new\\22_70_hv',
        'D:\\e04\\mrs\\gloc\\out_auto\\new\\22_71_hh','D:\\e04\\mrs\\gloc\\out_auto\\new\\22_71_hv')

if __name__=='__main__':

    for pro in range(0, 4):

        im1_p = loc_1[pro]
        im2_p = loc_2[pro]

        # im1_p = r'D:\e04\mrs\gloc\inp\246994611\scene_HV\22_76.tif'
        # im2_p = r'D:\e04\mrs\gloc\inp\246994611\scene_HV\22_76-ref.tif'

        out_path = out_[pro]
        print(im1_p)
        print(im2_p)
        print(out_path)

        dedode = KF.DeDoDe.from_pretrained(detector_weights='L-C4', descriptor_weights='G-C4').eval().to(device)

        adalam_config = KF.adalam.get_adalam_default_config()
        adalam_config["force_seed_mnn"] = False
        adalam_config["search_expansion"] = 16
        adalam_config["ransac_iters"] = 256
        adalam_config = {"device": device}

        fginn_config={
        'th':1,
        'spatial_th':30.0,
        'mutual':False
        }


        ##################################################################################################
        ###################################### INPUTS ##################################################################


        with rt.open(im1_p) as ima:
            x01, y01 = ima.bounds[0], ima.bounds[-1]
            xres1, yres1 = ima.transform.a, ima.transform.e

        with rt.open(im2_p) as imb:
            x02, y02 = imb.bounds[0], imb.bounds[-1]
            xres2, yres2 = imb.transform.a, imb.transform.e

        # im1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :, :2000, :2000]
        # im2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :, :2000, :2000]

        mode = 'combo'
        # mode = 'detector+descriptor'
        detector = dedode
        # desc = LD_mrd_blobhessian_sobel
        # matcher = lgm_matcher
        matcher = 'adalam'
        ransac_model = 'fundamental'
        # ransac_model = 'homography'
        # ran_list = ("fundamental", "homography")
        step2=2000 if pro<6 else 1512
        step1=2000 if pro<6 else 1407
        init=0
        device = K.utils.get_cuda_or_mps_device_if_available()
        for initx in np.arange(0, step1, 10000):
            for inity in np.arange(0,step2, 10000):
                try:
                    print('init='+str(initx)+' step='+str(step2))
                    # for detector in (comb_list):
                        # if detector._get_name() in (disk._get_name(),KND._get_name()):
                        #     print(True)
                    img1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :, initx:initx + step1,
                           inity:inity + step2] if pro < 6 else K.io.load_image(im1_p, K.io.ImageLoadType.RGB32,
                                                                              device=device)[None, :, initx:initx + step1,
                           inity:inity + step2]
                    img2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :, initx:initx + step1,
                           inity:inity + step2] if pro < 6 else K.io.load_image(im2_p, K.io.ImageLoadType.RGB32,
                                                                              device=device)[None, :, initx:initx + step1,
                           inity:inity + step2]

                    # else:
                    #     img1 = K.io.load_image(im1_p, K.io.ImageLoadType.GRAY32, device=device)[None, :, initx:initx + step1,
                    #           inity:inity + step2]
                    #     img2 = K.io.load_image(im2_p, K.io.ImageLoadType.GRAY32, device=device)[None, :, initx:initx + step1,
                    #           inity:inity + step2] if pro < 6 else K.io.load_image(im2_p, K.io.ImageLoadType.RGB32,
                    #                                                              device=device)[None, :, :, :]
                    im1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :,
                          initx:initx + step1,
                          inity:inity + step2] if pro < 6 else K.io.load_image(im1_p, K.io.ImageLoadType.GRAY32,
                                                                               device=device)[None, :, :, :]
                    im2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,
                          initx:initx + step1,
                          inity:inity + step2] if pro < 6 else K.io.load_image(im2_p, K.io.ImageLoadType.GRAY32,
                                                                               device=device)[None, :, :, :]
                    # for detector in (comb_list):
                    #     # desc = detector
                    #     im1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :, init:init + step1,
                    #            init:init+step2]
                    #     im2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,init:init+step1, init:init+step2] if pro<6 else K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,:, :]
                    #     if detector._get_name() == disk._get_name():
                    #         img1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :, init:init+step1, init:init+step2]
                    #         img2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,init:init+step1, init:init+step2] if pro<6 else K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,:, :]
                    #         # img2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :, :] #frs
                    #     elif detector._get_name() == dedode._get_name():
                    #         img1 = K.io.load_image(im1_p, K.io.ImageLoadType.RGB32, device=device)[None, :, init:init+step1, init:init+step2]
                    #         img2 = K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,init:init+step1, init:init+step2] if pro<6 else K.io.load_image(im2_p, K.io.ImageLoadType.RGB32, device=device)[None, :,:, :]
                    #     else:
                    #         img1 = K.io.load_image(im1_p, K.io.ImageLoadType.GRAY32, device=device)[None, :, init:init+step1, init:init+step2]
                    #         img2 = K.io.load_image(im2_p, K.io.ImageLoadType.GRAY32, device=device)[None, :,init:init+step1, init:init+step2] if pro<6 else K.io.load_image(im2_p, K.io.ImageLoadType.GRAY32, device=device)[None, :,:, :]

                    hw1 = th.tensor(img1.shape[2:],device=device)
                    hw2 = th.tensor(img2.shape[2:],device=device)
                    dedode_det_wts = ('L-C4', 'L-C4-v2')
                    dedode_des_wts = ('G-upright', 'G-C4')
                    # for desc in desc_list:
                    # try:
                    with ((th.inference_mode())):
                        try:

                            # for nf in (25, 50, 75, 100, 250, 500, 1000, 2000):
                            nf=50
                            for det in dedode_det_wts:
                                for des in dedode_des_wts:
                                    dedode = KF.DeDoDe.from_pretrained(detector_weights=det,
                                                                       descriptor_weights=des).eval().to(
                                        device)
                                    detector = dedode
                                    desc = dedode

                                    csv_path = (
                                            out_path + det + '_' + des + '_' + detector._get_name() + '_' + desc._get_name() +
                                            '_' + f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}" + '_'
                                            + f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_' + str(
                                        initx) + '_' + str(inity) + '_' + str(nf) + '1.csv')
                                    pngfile = (
                                            out_path + det + '_' + des + '_' + detector._get_name() + '_' + desc._get_name() +
                                            '_' + f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}"  + '_'
                                            + f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_' + str(
                                        initx) + '_' + str(inity) + '_' + str(nf) + '1.png')

                                    if not os.path.exists(csv_path) or not  os.path.exists(pngfile):
                                        print("Processing ...")
                                        inp = th.cat([img1, img2], dim=0)
                                        keypoints, scores, descriptors = detector(inp, n=nf)
                                        keypoints, descriptors = keypoints.to(device=device), descriptors.to(device=device)
                                        kps1, descs1 = keypoints[0], descriptors[0]
                                        kps2, descs2 = keypoints[1], descriptors[1]

                                        lafs1 = KF.laf_from_center_scale_ori(kps1[None], 96 * th.ones(1, len(kps1), 1, 1, device=device))
                                        lafs2 = KF.laf_from_center_scale_ori(kps2[None], 96 * th.ones(1, len(kps2), 1, 1, device=device))



                                        # for matcher in match_list:
                                        #     # try:
                                        #     if matcher=='adalam':

                                        dists, idxs = KF.match_adalam(
                                            descs1.squeeze(0),
                                            descs2.squeeze(0),
                                            lafs1,
                                            lafs2,
                                            config=adalam_config,
                                            hw1=hw1,
                                            hw2=hw2
                                        )

                                        print(f"{idxs.shape[0]} tentative matches with {detector._get_name()} {matcher._get_name() if matcher != 'adalam' else 'adalam'}")
                                        i1 = K.tensor_to_image(im1.cpu())
                                        i11 = i1 * 10 / np.max(i1)
                                        i2 = K.tensor_to_image(im2.cpu())
                                        i22 = i2 * 10 / np.max(i2)

                                        mkpts1, mkpts2 = get_matching_keypoints(lafs1, lafs2, idxs)
                                        mkpt1, mkpt2 = th.tensor(mkpts1).to(device), th.tensor(mkpts2).to(device)

                                        x1, y1 = zip(*mkpts1)
                                        x2, y2 = zip(*mkpts2)

                                        X1, Y1 = zip(*[xy_to_map(x1[i], y1[i], x01, y01, xres1, yres1, initx, inity) for i in range(0, len(x1))])
                                        X2, Y2 = zip(*[xy_to_map(x2[i], y2[i], x02, y02, xres2, yres2, initx, inity) for i in range(0, len(x2))])

                                        DIST =[np.sqrt(pow(X1[i]-X2[i], 2)+pow(Y1[i]-Y2[i], 2)) for i in range(0, len(x1))]
                                        dist = [np.sqrt(pow(x1[i]-x2[i], 2)+pow(y1[i]-y2[i], 2)) for i in range(0, len(x1))]


                                        with open(csv_path, mode='w', newline='') as file:
                                            writer = csv.writer(file)
                                            writer.writerow(['x1', 'y1', 'x2', 'y2', 'x1-map', 'y1-map', 'x2-map', 'y2-map', 'Distance', 'DISTS'])
                                            for i in range(0, len(X1)):
                                                writer.writerow([x1[i], y1[i], x2[i], y2[i], X1[i], Y1[i], X2[i], Y2[i], DIST[i], dists[i]])
                                        fig = plt.figure(figsize=(38.4, 21.6))

                                        manager = plt.get_current_fig_manager()

                                        draw_LAF_matches(
                                            lafs1.cpu(),
                                            lafs2.cpu(),
                                            idxs.cpu(),
                                            i11,
                                            i22,
                                            None,
                                            draw_dict={"inlier_color": (0.5, 1, 0.5),
                                                       "tentative_color": (0.8, 1, 0.8),
                                                       # "tentative_color": (1,0.1,0.2,0.6),
                                                       "feature_color": None, "vertical": False},
                                            fig=fig,
                                            return_fig_ax=True)

                                        plt.pause(1)
                                        plt.savefig(
                                            pngfile)
                                        # plt.close(fig)
                        except Exception as e:
                                    print("error is ", e)
                                    print(traceback.format_exc())
                except Exception as e:
                    print("error is ", e)
                    print(traceback.format_exc())
                    pass
                            #
                            # for ran in ran_list:
                            #     ransac_ = K.geometry.ransac.RANSAC(model_type=ran, inl_th=1, batch_size=2048, max_iter=1000, confidence=0.99) #### homography, fundamental
                            #     ransac = ransac_(mkpt1, mkpt2)
                            #     a=ransac_._get_name()
                            #     inliers=ransac[1]
                            #     # Fm, inliers = cv2.findFundamentalMat(
                            #     #     mkpts1, mkpts2, cv2.USAC_PARALLEL, 0.1, 0.999, 100000) ### USAC_MAGSAC,USAC_DEFAULT,USAC_PARALLEL,USAC_ACCURATE,USAC_PROSAC,USAC_FM_8PTS
                            #
                            #     # inliers = np.array([[1] if d<1.5 else [0] for d in DIST])
                            #     # inliers = np.array([[1] if d<2*xres else [0] for d in DIST])
                            #     inliers = inliers > 0
                            #
                            #     print(f"{inliers.sum()} inliers with {detector._get_name()} {matcher._get_name() if matcher != 'adalam' else 'adalam'}")
                            #

                            #
                            #     if detector in mrd_detectors:
                            #         det_text = detector.model._get_name()
                            #         if detector != MRD_kn:
                            #             det_text = detector.model._get_name() + '_'  + detector.model.grads_mode
                            #     elif detector in ssd_detectors:
                            #         det_text = detector.resp._get_name() + '_' + detector.resp.grads_mode
                            #     else:
                            #         det_text = detector.model._get_name()
                                # out_param = out_path + detector._get_name() + '_' + det_text + '__'+ desc._get_name()+ f"{desc.descriptor._get_name() if desc in ld_desc else ''}"
                                # +'_'+ f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}" + f"{str(matcher.th) if matcher in th_matcher else ''}"+'_'
                                # +f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_'+ransac_._get_name()+'_' +ransac_.model_type+'_'+str(ransac_.inl_th)+'_'+str(init)+'-'+str(init+step)
                                #
                                # if not (os.path.exists(out_param+'.png') or os.path.exists(out_param+'.csv')):

                                # plt.savefig(out_path + detector._get_name() + '_' + det_text + '__'+ desc._get_name()+ f"{desc.descriptor._get_name() if desc in ld_desc else ''}"
                                # +'_'+ f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}" + f"{str(matcher.th) if matcher in th_matcher else ''}"+'_'
                                # +f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_'+ransac_._get_name()+'_' +ransac_.model_type+'_'+str(ransac_.inl_th)+'_'+str(init)+'-'+str(init+step) +'.png', bbox_inches='tight', format='png' )
        #
        #                         csv_path = (
        #                                     out_path + detector._get_name() + '_' + det_text + '__' + detector._get_name() + '_'
        #                                     + f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}" + f"{str(matcher.th) if matcher in th_matcher else ''}"
        #                                     + '_' + f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_' + ransac_._get_name() + '_' + ransac_.model_type + '_' + str(
        #                                 ransac_.inl_th) + '_' + str(init) + '-' + str(init + step) + '.csv')
        #                         # csv_path = (out_path + detector._get_name() + '_' + det_text + '__'+ desc._get_name()+ f"{desc.descriptor._get_name() if desc in ld_desc else ''}"+'_'
        #                         #             + f"{matcher._get_name() if matcher != 'adalam' else 'adalam'}" +f"{str(matcher.th) if matcher in th_matcher else ''}"
        #                         #             +'_'+f"{matcher.match_mode if matcher != 'adalam' else 'adalam'}" + '_'+ransac_._get_name()+'_'+ransac_.model_type+'_'+str(ransac_.inl_th)+'_'+str(init)+'-'+str(init+step) +'.csv')
        #                         with open(csv_path, mode='w', newline='') as file:
        #                             writer = csv.writer(file)
        #                             # writer.writerow(['x1', 'y1', 'x2', 'y2','Distance', 'inliers'])
        #                             # for i in range(0, len(x1)):
        #                                 # writer.writerow([x1[i], y1[i], x2[i], y2[i], DIST[i], inliers[:,0][i]])
        #                             writer.writerow(['x1', 'y1', 'x2', 'y2', 'x1-map', 'y1-map', 'x2-map', 'y2-map', 'Distance', 'inliers'])
        #                             for i in range(0, len(X1)):
        #                                 writer.writerow([x1[i], y1[i], x2[i], y2[i], X1[i], Y1[i], X2[i], Y2[i], DIST[i], inliers[i]])
        #                         plt.close(fig)
        #
        #
        #
        #                 except Exception as e:
        #                     print('bankai errpor is ',e)
        #     except Exception as e:
        #         print('shikai error is ',e)
        # # init = init+step1