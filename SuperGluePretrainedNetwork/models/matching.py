# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import numpy as np

from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})

            if 'valid0' in data:
                new_pred0 = {}

                for key in pred0.keys():

                    if key == "descriptors":
                        continue

                    v = pred0[key]

                    device = v[0].get_device()
                    original_elem = v[0].cpu()

                    new_elem = []

                    for i in range(original_elem.size(dim=0)):
                        if data["valid0"][i]:
                            new_elem.append(original_elem[i])
                    
                    new_v = []
                    new_v.append(torch.as_tensor(np.array(new_elem)))
                    new_v[0] = new_v[0].to(device)
                    new_pred0[key+'0'] = new_v

                v = pred0["descriptors"]

                device = v[0].get_device()
                original_elem = v[0].cpu()

                new_elem = []

                for i in range(original_elem.size(dim=1)):
                    if data["valid0"][i]:
                        new_elem.append(original_elem[:, i])

                new_v = []
                new_v.append(torch.as_tensor(np.array(new_elem).T))
                new_v[0] = new_v[0].to(device)
                new_pred0['descriptors0'] = new_v

                pred = {**pred, **new_pred0}
            else:
                pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            if 'valid0' in data:
                new_pred1 = {}

                for key in pred1.keys():

                    if key == "descriptors":
                        continue

                    v = pred1[key]

                    device = v[0].get_device()
                    original_elem = v[0].cpu()

                    new_elem = []

                    for i in range(original_elem.size(dim=0)):
                        if data["valid1"][i]:
                            new_elem.append(original_elem[i])

                    new_v = []
                    new_v.append(torch.as_tensor(np.array(new_elem)))
                    new_v[0] = new_v[0].to(device)
                    new_pred1[key+'1'] = new_v

                v = pred1["descriptors"]

                device = v[0].get_device()
                original_elem = v[0].cpu()

                new_elem = []

                for i in range(original_elem.size(dim=1)):
                    if data["valid1"][i]:
                        new_elem.append(original_elem[:, i])

                new_v = []
                new_v.append(torch.as_tensor(np.array(new_elem).T))
                new_v[0] = new_v[0].to(device)
                new_pred1['descriptors1'] = new_v


                pred = {**pred, **new_pred1}
            else:
                pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
            # pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred
