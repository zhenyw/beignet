/*
 * Copyright © 2012 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 */

/**
 * \file image.hpp
 *
 */
#ifndef __GBE_IR_IMAGE_HPP__
#define __GBE_IR_IMAGE_HPP__

#include "ir/register.hpp"
#include "sys/map.hpp"

extern "C" {
  struct ImageInfo;
}

namespace gbe {
namespace ir {

  class Context;
  /*! An image set is a set of images which are defined in kernel args.
   *  We use this set to gather the images here and allocate a unique index
   *  for each individual image. And that individual image could be used
   *  at backend to identify this image's location.
   */
  class ImageSet
  {
  public:
    /*! Append an image argument. */
    void append(Register imageReg, Context *ctx);
    /*! Get the image's index(actual location). */
    const uint32_t getIdx(const Register imageReg) const;
    size_t getDataSize(void) { return regMap.size(); }
    size_t getDataSize(void) const { return regMap.size(); }
    void getData(struct ImageInfo *imageInfos) const;
    void operator = (const ImageSet& other) {
      regMap.insert(other.regMap.begin(), other.regMap.end());
    }
    ImageSet(const ImageSet& other) : regMap(other.regMap.begin(), other.regMap.end()) { }
    ImageSet() {}
    ~ImageSet();
  private:
    map<Register, struct ImageInfo *> regMap;
    GBE_CLASS(ImageSet);
  };
} /* namespace ir */
} /* namespace gbe */

#endif /* __GBE_IR_IMAGE_HPP__ */